#!/usr/bin/env python3
"""What a forward binds and the store list does not hold - both directions.

**The deployment list must hold every shape any walk of any stage has produced.** A walk is one
deployment's rows at one element type, so a file that serves several deployments holds each shape at
more than one type - and the failure that matters is the shape that is absent at *every* type, because
that key cannot be built by any run. `--by-shape` compares walks and file with the element type
normalised to `<t>` and takes any number of `--walk` files, which is the union rule: the union of the
walks each stage has produced, against the file. The other half of the rule - rows no walk states but
the construction requires - is `check_manifests.py`'s question, not this one's.

The store a caller builds comes from a fixture under `notes/programs/`, one per deployment, and it is
a **walk's rows**: `cargo run -p minimax-h3 --example programs -- <root> --dtype <t> ...` prints them
from the model's own layers, and `h3_fixture --walk <print> --commit <rev>` writes the file with the
tree it came from. Three checks read that file, and they answer three different questions:

* `key_coverage` asks whether a **listed** key is buildable for a target;
* `check_manifests.py` asks whether a **descriptor the code derives** has a manifest cell;
* this one asks whether a **key a forward binds** is in the list.

The first cannot see a bound key that is not listed, and the second cannot see it either unless the
descriptor walk knows about it - so a store can be complete over everything anyone wrote down and
still stop a device run one key later. That is what happened twice (`logits_gather`, then `gemm`
`epilogue_bias`).

The walk file comes from a forward that reports its launches. `ops::capture::Sink` is that recorder:
every model layer launches through `Forward::operate`, which reports `(operator, key)` to the sink
installed on the thread, so a harness installs one around the forward and writes what it recorded.
One line per launch, `operator<TAB>key`, in launch order - duplicates are fine and are counted.

Usage:

    walk_gaps.py --walk artifacts/h3-stage1/walk.txt \\
                 --fixture notes/programs/minimax-h3-keys.txt [--unbound]

Exit status: 1 when a bound key has no fixture line (the run would stop with `no directory`/`no
kernel`), 0 otherwise. Unbound fixture lines are printed but never failed on by default.

**A file is one deployment's rows, so an unbound line means one of two things, and they are worth
telling apart before it is dismissed:** the fixture is another deployment's (a `--dtype` or a degree
that does not match this walk), or the walk is narrower than the file because a component it cannot
reach - a stage that has never run - contributed its rows to a *bootstrap* list and not to a walk. The
first is a wrong file; the second is a component this run never exercised, and neither is "normal
coverage". `--allow-unbound` silences the count entirely, so it is for a reader who has already said
which of the two it is.
"""

import argparse
import sys
from collections import Counter


def read_lines(path):
    """`(operator, key)` pairs of one file, and how many times each appears."""
    pairs = Counter()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split("\t")
            if len(fields) != 2:
                raise SystemExit(
                    f"{path}:{number}: expected `operator<TAB>key`, found {text!r}"
                )
            pairs[(fields[0], fields[1])] += 1
    return pairs


ELEMENTS = ("bf16", "f16", "f32", "f64")


def shape_of(operator, key):
    """One key with its element types normalised, so the same shape compares across deployments."""
    for element in ELEMENTS:
        key = key.replace(element, "<t>")
    return (operator, key)


def types_of(key):
    """The element types a key names, in the order they appear."""
    return [element for element in ELEMENTS if element in key]


# The operators whose key is a function of the request's raster. A walk and
# a fixture that disagree about the canvas describe two deployments, and a
# difference in these rows is that disagreement, not a missing row: `frame_pad`
# takes `min(ROW_PIXELS, next power of two of the output's columns)` pixels per
# program, and the `scaled_add` widths are latent extents at the canvas's grid.
CANVAS_CLASS = ("frame_pad", "scaled_add")


def statements(path):
    """What a file says about the raster it describes: `(rasters, text)`.

    A walk states `raster sizes: [(w, h), ...]` in its header. Reading the
    statements rather than counting shapes is the point: a canvas-class row
    difference cannot be decided without knowing both sides' canvas.
    """
    rasters, text = [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            stripped = line.strip()
            if "raster sizes:" in stripped:
                inner = stripped.split("raster sizes:", 1)[1].split("]", 1)[0]
                rasters = [
                    tuple(int(part) for part in pair.strip(" ()").split(","))
                    for pair in inner.strip(" [").split("),")
                    if pair.strip(" ()")
                ]
            elif stripped.lower().startswith("# raster:"):
                text = stripped.split(":", 1)[1].strip()
    return rasters, text


def by_shape(args):
    """Every shape the walks produced, against the shapes the file holds at any element type."""
    shapes = {}
    total = 0
    for path in args.walk:
        walked = read_lines(path)
        total += sum(walked.values())
        for operator, key in walked:
            shapes.setdefault(shape_of(operator, key), {}).setdefault(
                path, set()
            ).update(types_of(key))
    fixture = read_lines(args.fixture)
    held = {shape_of(operator, key) for operator, key in fixture}

    print(
        f"walks: {len(args.walk)} file(s), {total} launches, {len(shapes)} shapes"
    )
    for path in args.walk:
        print(f"  {path}")
    print(f"fixture: {len(fixture)} keys, {len(held)} shapes")
    # Both sides' statements about the canvas, printed before the count, so a
    # reader meets the comparison's scope before its answer.
    walked_rasters = sorted(
        {raster for path in args.walk for raster in statements(path)[0]}
    )
    stated = next(
        (
            text
            for path in args.walk
            if (text := statements(path)[1]) is not None
        ),
        None,
    )
    fixture_text = statements(args.fixture)[1]
    print(f"walk rasters: {walked_rasters or 'unstated'}")
    print(f"walk statement: {stated or 'none'}")
    print(f"fixture statement: {fixture_text or 'none'}")

    missing = sorted(set(shapes) - held)
    canvas = [shape for shape in missing if shape[0] in CANVAS_CLASS]
    for operator, key in missing:
        where = sorted(shapes[(operator, key)])
        launched_at = sorted(
            {
                element
                for path in where
                for element in shapes[(operator, key)][path]
            }
        )
        mark = (
            "  RASTER-DEPENDENT"
            if operator in CANVAS_CLASS
            else "  MISSING at every type"
        )
        print(f"{mark}: {operator}\t{key}")
        print(f"    launched at {', '.join(launched_at)} by {', '.join(where)}")
    if canvas:
        fixture_says = fixture_text or "nothing"
        print(
            f"\n{len(canvas)} of these are {', '.join(CANVAS_CLASS)} rows, whose key"
            " is the request's raster. The walk states "
            f"{walked_rasters or 'no raster'}; the store list states {fixture_says}."
            " A difference here is a disagreement about the canvas, not a"
            " missing row: a store built from the file serves the canvas the"
            " file describes."
        )
    if missing:
        print(
            f"\n{len(missing)} shape(s) a walk produced and the file cannot build at any element "
            "type; "
            f"{len(missing) - len(canvas)} of them at any raster: each of those stops a run at "
            "`no kernel`"
        )
        return 1
    print(
        "\nevery shape any walk produced is in the fixture, at some element type"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--walk",
        required=True,
        action="append",
        help="what the forward reported, one launch a line; repeat it to take a union of stages",
    )
    parser.add_argument(
        "--by-shape",
        action="store_true",
        help="compare with the element type normalised, so a walk of one deployment checks the "
        "shapes a file serving several deployments must hold; fails on a shape absent at every type",
    )
    parser.add_argument(
        "--fixture", required=True, help="the store list the caller builds from"
    )
    parser.add_argument(
        "--unbound",
        action="store_true",
        help="list every fixture line no launch bound (the normal case for one deployment, so the "
        "default prints the count and a few)",
    )
    args = parser.parse_args(argv)

    if args.by_shape:
        return by_shape(args)
    walk = read_lines(args.walk[0])
    fixture = read_lines(args.fixture)

    bound = sorted(walk)
    unbound = sorted(set(fixture) - set(walk))
    missing = sorted(set(walk) - set(fixture))

    calls = sum(walk.values())
    print(f"walk: {len(bound)} distinct keys over {calls} launches")
    print(f"fixture: {len(fixture)} keys")
    for operator, key in missing:
        print(
            f"  MISSING from the fixture: {operator}\t{key}  ({walk[(operator, key)]} launches)"
        )
    if unbound:
        print(
            f"fixture lines no launch bound ({len(unbound)}; normal for one deployment):"
        )
        shown = unbound if args.unbound else unbound[:8]
        for operator, key in shown:
            print(f"  unbound: {operator}\t{key}")
        if len(shown) < len(unbound):
            print(f"  … {len(unbound) - len(shown)} more with --unbound")

    if missing:
        print(
            f"\n{len(missing)} key(s) a forward binds and the store list does not hold"
        )
        return 1
    print("\nevery key the walk bound is in the fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
