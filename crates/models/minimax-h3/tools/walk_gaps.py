#!/usr/bin/env python3
"""What a forward binds and the store list does not hold - both directions.

**The deployment list must hold every shape any walk of any stage has produced.** A walk is one
deployment's rows at one element type, so a file that serves several deployments holds each shape at
more than one type - and the failure that matters is the shape that is absent at *every* type, because
that key cannot be built by any run. `--by-shape` compares walks and file with the element type
normalised to `<t>` and takes any number of `--walk` files, which is the union rule: the union of the
walks each stage has produced, against the file. **Run it twice at the deployment's own element type, once conditioned and once not, and
take the union.** An unconditioned walk never reaches the audio encoder or the video decoder,
and a walk at one element type cannot show the rows of a stack that runs at another; each
walk of this tree states 136 launches and the two together name shapes neither names alone.
`--walk` is repeatable for exactly this reason.

**And read each component's rows from the walk whose element type is that component's.** A
walk keys every component at one type (`--dtype` is the example's, not the deployment's), while
a served deployment has the stack at BF16 and the video VAE at the checkpoint's own FP16; so the
pair gives each *shape* at two types and only one of them is the type that shape is launched at.

**A gap is a defect only when the walk and the file were asked the same question**, and
a question has a raster, a clip length, a token count and an element count. The walk's own
header states those, so the check prints them and answers a gap in three disjoint ways: a row
the file lacks (a run stops at `no kernel`), a raster-class row (`frame_pad`, `scaled_add`),
or a number under an extent field that equals one of the walk's own. **A block, a head width
and a pixel tile are program constants; a column count at a token length is the request's** -
which is why only `columns`, `elements`, `rows`, `extent`, `samples` and `tokens` are read as
extents, and why a clip length of 256 does not excuse `head_dim_256`.

The other half of the rule - rows no walk states but
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
import re
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


def sample_of(path):
    """The sample a walk was generated at, as `{dimension: [numbers]}`.

    A walk's header states the question it was asked: clip lengths, rasters, prompt
    lengths, packed rows, trajectory elements, audio tracks. A key carrying one of
    those numbers is a key the request decides, so a gap in it is a different
    question asked of the file rather than a row the file lacks - and reading the
    statement is what tells the two apart without a hand-written list of which
    operators vary.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if "clip lengths:" not in line and "raster sizes:" not in line:
                continue
            dimensions = {}
            for name, value in re.findall(
                r"([A-Za-z][A-Za-z ]*):\s*(\[[^\]]*\])", line
            ):
                numbers = [
                    int(part) for part in re.split(r"\D+", value) if part
                ]
                if numbers:
                    dimensions[name.strip()] = numbers
            return dimensions
    return {}


# The key fields that name an extent the request decides: a column count, a row count, an
# element count, a table's rows, a fade's extent. A number under one of these that equals a
# number in the walk's own header is the walk's question, not the file's answer - while the
# same number under `block`, `head_dim`, `block_pixels` or `tile` is a program constant, and
# reading only these fields is what keeps a clip length from matching a head width.
EXTENT_FIELDS = ("columns", "elements", "rows", "extent", "samples", "tokens")


def extents_in(key):
    """The numbers a key states under one of `EXTENT_FIELDS`, with the field's name."""
    found = []
    for field in EXTENT_FIELDS:
        for match in re.finditer(rf"(?:^|_){field}_(\d+)", key):
            found.append((field, int(match.group(1))))
    return found


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

    # The walk's own sample: a gap is a defect only when both sides were asked the
    # same question, so the question is printed beside the answer.
    sample = {}
    for path in args.walk:
        for name, values in sample_of(path).items():
            sample.setdefault(name, sorted(set(values)))
    if sample:
        print(
            "walk sample: "
            + ", ".join(
                f"{name} {values}" for name, values in sorted(sample.items())
            )
        )
    sampled = {
        number: name for name, values in sample.items() for number in values
    }

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
        carried = sorted(
            {
                (field, number)
                for field, number in extents_in(key)
                if number in sampled
            }
        )
        if carried:
            asked = ", ".join(
                f"{field} {number} = the walk's {sampled[number]}"
                for field, number in carried
            )
            mark = f"  SAMPLE-DEPENDENT ({asked})"
        elif operator in CANVAS_CLASS:
            mark = "  RASTER-DEPENDENT"
        else:
            # The element types the walk launched it at, in the verdict line: the comparison
            # normalises the type away, so the one field a row is *written* with is the one the
            # count lost - a shape missing at every type is a row to add, but only a reader who
            # sees `f16` knows which row.
            mark = (
                f"  MISSING at every type, launched at {', '.join(launched_at)}"
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
        # Three answers, not one: a row the file lacks, a row the *request* decides, and - where
        # the walk states no sample - a difference nobody can classify yet.
        # A shape can be both: `scaled_add ... columns_4096` is a canvas-class operator whose
        # width is the walk's prompt length. The sample answers it first, since it is the more
        # specific claim - and a shape counted twice would make the remainder too small.
        dependent = sorted(
            shape
            for shape in missing
            if any(number in sampled for _, number in extents_in(shape[1]))
        )
        canvas = sorted(set(canvas) - set(dependent))
        real = len(missing) - len(canvas) - len(dependent)
        print(
            f"\n{len(missing)} shape(s) a walk produced and the file cannot build at any element "
            f"type: {real} at any raster (each stops a run at `no kernel`), "
            f"{len(canvas)} raster-class, {len(dependent)} the walk's own sample"
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
