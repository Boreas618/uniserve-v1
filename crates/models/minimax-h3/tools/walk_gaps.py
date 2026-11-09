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

The store a caller builds comes from a key file, one per deployment, and it is
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

    walk_gaps.py --walk <walk file> --fixture <key file> [--unbound]

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
import pathlib
import re
import subprocess
import sys
import tempfile
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


def clips(path):
    """The clip lengths a walk states it ran at.

    A width that is an exact multiple of the walk's clip length is derived from that length - the
    audio decoder multiplies it by each stage's rate, the postprocessor's `scaled_add` covers a
    whole frame - so the multiple is a fact about the derivation and not about the list. Printing
    it stops such a width being read as a canvas's or a request's.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if "clip lengths:" in line:
                inner = line.split("clip lengths:", 1)[1].split("]", 1)[0]
                return [
                    int(value)
                    for value in inner.strip(" [").split(",")
                    if value.strip()
                ]
    return []


LADDER = re.compile(r"clip ladder \[(?P<list>[^\]]*)\]")


def ladder(path):
    """The clip lengths a file states it serves, from its own `axes:` line.

    A row set is stated for a set of clips. A walk at a clip outside that set produces
    audio rows for a length no row set covers - so the difference between the sides is a
    difference of deployment, and the comparison cannot separate it from a gap.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            found = LADDER.search(line)
            if found:
                return [
                    int(value)
                    for value in found["list"].split(",")
                    if value.strip()
                ]
    return []


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
                # Two forms are written today: a bracketed list of tuples,
                # `[(672, 1536), (768, 1344)]`, and the same list quoted as one string,
                # `"(672, 1536), (768, 1344)`. The quoted form is the writer's defect, but a
                # reader that crashes on it reports a broken harness rather than a broken
                # header, so both are read and the difference is left to the writer's check.
                inner = stripped.split("raster sizes:", 1)[1].lstrip()
                # The list is the rest of the field, so it ends where the next one begins: at
                # the closing quote of the quoted form, at `]` of the bracketed form, or at
                # the `, packed rows:` that follows it in the header's one-line field list.
                if inner.startswith(('"', "'")):
                    quote = inner[0]
                    inner = inner[1:].split(quote, 1)[0]
                else:
                    inner = inner.split("]", 1)[0]
                inner = inner.split(", packed rows:", 1)[0].replace('"', "").replace("'", "")
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


def classify(missing, shapes, rasters_of, sampled, held=()):
    """Split the missing shapes by the question each absence answers.

    Four answers, not one: a row the file lacks at every raster, a row the request's raster
    decides, a row the walk's own sample decides, and a row the file *holds* at another clip
    length. A shape can be several of these - `scaled_add ... columns_4096` is a canvas-class
    operator whose width is the walk's prompt length - and the sample answers first, because it
    is the more specific claim: a shape counted twice would make the remainder too small.

    **`frames differing` is not `missing`.** A program the file holds with `extent_124` and a
    walk that launches with `extent_362` is one program at two clip lengths: the file needs a
    row, but calling it a row the file lacks at every raster reads as a program no store can
    build, and a run then stops at `no kernel` for a row `extent` alone decides. The program is
    the key with its extent fields removed, so the match is on that and on nothing else.

    **A canvas-class operator is canvas-class only when the walks that produced it share one
    raster.** With a single raster the label absorbs every absence that does not depend on the
    raster - twelve rows a store must hold were filed under it that way - so the absence has to
    survive a change of canvas before the canvas is allowed to explain it.
    """
    # The programs the file holds, by the key with its extent fields removed: a clip length reaches
    # `extent` alone, so two rows of one program differ in exactly these numbers.
    programs = {
        (operator, without(key, EXTENTS)) for operator, key in held
    }
    canvas, dependent, frames, real = [], [], [], []
    for shape in missing:
        carried = sorted(
            (field, number)
            for field, number in extents_in(shape[1])
            if number in sampled
        )
        if carried:
            dependent.append(shape)
        elif (
            shape[0] in CANVAS_CLASS
            and len({tuple(rasters_of[path]) for path in shapes[shape]}) < 2
        ):
            canvas.append(shape)
        elif (shape[0], without(shape[1], EXTENTS)) in programs:
            frames.append(shape)
        else:
            real.append(shape)
    return canvas, dependent, frames, real


# The extents a key carries: a launch decides these, so counting them apart from the rest of the
# key says how much a single request multiplies the rows of one operator.
EXTENTS = ("columns", "elements", "rows", "extent", "samples", "tokens")


class _Args:
    """The two fields `wellformed` reads, so the self-test can drive it without a parser."""

    def __init__(self, fixture):
        self.fixture = fixture


def without(key, fields):
    """The key with each named field and its value removed.

    A key is `field_value_field_value_...`, so a field takes the token after it. What remains names
    the program a launch asks for; what was removed names the run that asked.
    """
    parts, index = [], 0
    tokens = key.split("_")
    while index < len(tokens):
        if tokens[index] in fields:
            index += 2
            continue
        parts.append(tokens[index])
        index += 1
    return "_".join(parts)


def families(args):
    """Per operator: distinct keys launched, distinct keys with `--fields` removed, and the ratio.

    A field that leaves a key multiplies an operator's rows by the number of values a legitimate run
    produces for it. That ratio is a fact about launches - it needs no store and no device - so it
    answers "how many of these rows exist because of this field", not "which of them differ in the
    compiled program". The second question needs two built keys and their cubins.
    """
    fields = set(args.fields.split(","))
    keys = {}
    for path in args.walk:
        for operator, key in read_lines(path):
            keys.setdefault(operator, set()).add(key)
    print(
        f"{len(args.walk)} walk(s), {len(keys)} operators, removing {', '.join(sorted(fields))}"
    )
    total, collapsed = 0, 0
    for operator in sorted(keys):
        whole = keys[operator]
        fewer = {without(key, fields) for key in whole}
        total += len(whole)
        collapsed += len(fewer)
        if len(whole) != len(fewer):
            print(
                f"  {operator}: {len(whole)} key(s) -> {len(fewer)} without the field(s)"
            )
    print(f"{total} key(s) launched, {collapsed} with the field(s) removed")


#: The operators' sources, found from this file's own place in the repository.
OPS = pathlib.Path(__file__).resolve().parents[4] / "crates/ops/src"
NAME = re.compile(r'const NAME: &\'static str = "([a-z0-9_]+)"')
COUNT = re.compile(r"^# (\d+) data rows(?:, (\d+) operators)?")


def wellformed(args):
    """The list's own properties: its stated count, its uniqueness, and its operator names.

    Three failures that a walk comparison cannot see, because each makes the *file* wrong rather
    than a row missing: a header that states a count the file does not hold (the count every later
    reader quotes), a line repeated (the same program compiled twice, or a collision hidden), and an
    operator name `crates/ops` does not define (an install failure far from here).
    """
    text = pathlib.Path(args.fixture).read_text().splitlines()
    rows, stated, stated_ops = [], None, None
    for line in text:
        if line.startswith("#"):
            found = COUNT.match(line)
            if found:
                stated = int(found.group(1))
                # The same line usually names the operators; when it does, it is a
                # second statement about the file and drifts on its own.
                stated_ops = int(found.group(2)) if found.group(2) else None
            continue
        if "\t" in line:
            operator, key = line.split("\t", 1)
            rows.append((operator, key))
    failures = 0
    if stated is None:
        print(
            "the header states no row count: a reader quoting the file has no number"
        )
        failures += 1
    elif stated != len(rows):
        print(
            f"the header states {stated} data rows and the file holds {len(rows)}"
        )
        failures += 1
    operators = {operator for operator, _ in rows}
    if stated_ops is None:
        print(
            "the header states no operator count: the line naming the rows should"
            " name how many operators they come from"
        )
        failures += 1
    elif stated_ops != len(operators):
        print(
            f"the header states {stated_ops} operators and the file holds"
            f" {len(operators)}"
        )
        failures += 1
    repeats = sorted({row for row in rows if rows.count(row) > 1})
    if repeats:
        print(f"{len(repeats)} line(s) repeated:")
        for operator, key in repeats[:5]:
            print(f"  {operator}\t{key}")
        failures += 1
    defined = set()
    for path in OPS.rglob("*.rs"):
        defined.update(NAME.findall(path.read_text()))
    unknown = sorted({operator for operator, _ in rows} - defined)
    if unknown:
        print(
            f"{len(unknown)} operator(s) `crates/ops` does not define: {unknown}"
        )
        failures += 1
    if failures:
        return 1
    print(
        f"the list states what it holds: {len(rows)} rows over"
        f" {len(operators)} operators, no repeat, and every operator among the"
        f" {len(defined)} `crates/ops` defines"
    )
    return 0


def require_rows(walks):
    """Refuse a walk with no rows: every comparison here passes if a side is empty."""
    empty = [path for path in walks if not read_lines(path)]
    if empty:
        raise SystemExit(
            f"{', '.join(empty)}: a walk with no rows proves nothing - every"
            " comparison here passes when one side is empty, so the run that"
            " produced this file failed before it printed a row"
        )


def phantom(args):
    """Operators the list holds and no walk in the set launches, with their row counts.

    This is not "no launcher exists" - a walk reaches only what its flags reach. It is the list of
    operators where the two instruments have nothing to say to each other, which is where a row set
    written from the checkpoint's module names drifts from what the model builds: `snake_beta` is
    seven such rows, and the walk's own activation row is the fused `audio_activation`.
    """
    require_rows(args.walk)
    walked = set()
    for path in args.walk:
        walked.update(operator for operator, _ in read_lines(path))
    fixture = read_lines(args.fixture)
    counts = {}
    for operator, _ in fixture:
        counts[operator] = counts.get(operator, 0) + 1
    silent = sorted(op for op in counts if op not in walked)
    for operator in silent:
        print(
            f"  no walk launches: {operator}  ({counts[operator]} row(s) in the list)"
        )
    if not silent:
        print(
            "every operator the list holds is launched by some walk in this set"
        )
        return 0
    print(
        f"\n{len(silent)} operator(s), {sum(counts[op] for op in silent)} row(s),"
        " the list holds and these walks never launch. A walk reaches what its flags"
        " reach, so this names a disagreement between the two instruments - not a"
        " launcher's absence, which only the model's own source can say."
    )
    return 1


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
    # An empty walk is not a walk. Every lane below compares the two sides, and a
    # side with no rows makes all of them pass on any list - the same vacuity the
    # canvas union check refuses when its parts stop differing.
    require_rows(args.walk)
    # A walk at a clip the file does not serve is a different deployment, not a set of
    # gaps: its audio rows are for a length no row set covers, and their widths would
    # read as missing rows. Refused rather than classified.
    served = ladder(args.fixture)
    outside = sorted(
        {
            clip
            for path in args.walk
            for clip in clips(path)
            if served and clip not in served
        }
    )
    if outside:
        raise SystemExit(
            f"the walks state clip(s) {outside} and the file states {served}: a walk "
            "at a clip the file does not serve produces audio rows for a length no row "
            "set covers, so this comparison cannot separate a gap from a different "
            "deployment of the same model"
        )
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
    # A width an audio operator carries is derived from the track through the crate's stage
    # arithmetic, which this tool does not follow: the number a key lands on is not the number on
    # the command line (`--track 477166` produces 2985 x 160 = 477600, not 477166). So an audio
    # width is sample-dependent by construction, and the sentence says so rather than letting the
    # per-shape mark imply the tool checked a derivation it cannot see.
    tracks = list(sample.get("audio tracks", []))
    if tracks:
        print(
            f"the walks state an audio track {tracks}: an audio width is derived from"
            " it through the crate's stages, and this tool does not follow that"
            " arithmetic - an audio `scaled_add`/`scaled_residual` gap is"
            " sample-dependent by construction, not a row the file lacks"
        )

    missing = sorted(set(shapes) - held)
    # The rasters each walk states, so a shape absent from walks of two *different* canvases is not
    # called canvas-class: the canvas cannot explain an absence that survives a change of canvas.
    stated = {path: statements(path)[0] for path in args.walk}
    canvas, dependent, frames, real = classify(missing, shapes, stated, sampled, held)
    for operator, key in missing:
        shape = (operator, key)
        where = sorted(shapes[shape])
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
        if shape in dependent:
            asked = ", ".join(
                f"{field} {number} = the walk's {sampled[number]}"
                for field, number in carried
            )
            mark = f"  SAMPLE-DEPENDENT ({asked})"
        elif shape in canvas:
            mark = "  RASTER-DEPENDENT"
        elif operator in CANVAS_CLASS:
            mark = (
                "  MISSING at every raster walked (the canvas cannot explain it:"
                " absent at two or more)"
            )
        else:
            # The element types the walk launched it at, in the verdict line: the comparison
            # normalises the type away, so the one field a row is *written* with is the one the
            # count lost - a shape missing at every type is a row to add, but only a reader who
            # sees `f16` knows which row.
            mark = (
                f"  MISSING at every type, launched at {', '.join(launched_at)}"
            )
        lengths = sorted({clip for path in where for clip in clips(path)})
        multiples = [
            f"{number} = clip {clip} x {number // clip}"
            for field, number in extents_in(key)
            if field in ("columns", "elements")
            for clip in lengths
            if number % clip == 0 and number != clip
        ]
        note = f"  [{'; '.join(multiples)}]" if multiples else ""
        print(f"{mark}: {operator}\t{key}{note}")
        print(f"    launched at {', '.join(launched_at)} by {', '.join(where)}")
    if frames:
        print(
            f"\n{len(frames)} of these are programs the file holds at another clip length: its"
            " `extent` is the frame count, so the entry it needs is a row of the same program"
            " at the walk's own `extent` - not a program no store can build. A run that stops"
            " at `no kernel` for one of these is stopped by the clip it was asked for."
        )
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
        print(
            f"\n{len(missing)} shape(s) a walk produced and the file cannot build at any element "
            f"type: {len(real)} at any raster (each stops a run at `no kernel`), "
            f"{len(frames)} at a clip length the file states another way, "
            f"{len(canvas)} raster-class, {len(dependent)} the walk's own sample"
        )
        return 1
    print(
        "\nevery shape any walk produced is in the fixture, at some element type"
    )
    return 0


def self_test() -> int:
    """Prove the classification: five absences, one answer each.

    The regression this guards is the one that made `0 at any raster` wrong: a canvas-class
    operator absent from walks of two different canvases is a row a store must hold, and the
    operator's name alone must never be enough to file it under the canvas.
    """

    def key(columns: int) -> str:
        return f"dtype_f32_columns_{columns}_block_1024"

    canvas_row = ("frame_pad", "dtype_f32_block_pixels_64_block_channels_32")
    sample_row = ("scaled_residual", key(123456))
    twice_row = ("scaled_add", key(131072))
    once_row = ("scaled_add", key(4096))
    # One program the file holds at 124 frames and a walk that launches at 362: the row is a
    # clip-length difference, and calling it a row the file lacks reads as a program no store can
    # build. The operator and the key minus its extents are the whole of the match.
    clip_missing = ("video_vae", "dtype_f32_extent_362_axis_2")
    clip_held = ("video_vae", "dtype_f32_extent_124_axis_2")
    # `scaled_add ... 4096` is also the walk's prompt length, so the sample answers it first:
    # the ordering in `classify` is part of what this test states.
    # `a` and `b` are two walks at the same canvas - the conditioned and the plain one a caller
    # runs - and `c` is the second canvas whose presence makes an absence a fact.
    shapes = {
        canvas_row: {"a": {"f16"}, "b": {"f16"}},
        sample_row: {"a": {"f32"}},
        twice_row: {"a": {"bf16"}, "c": {"bf16"}},
        once_row: {"a": {"bf16"}},
        clip_missing: {"a": {"f32"}},
    }
    rasters_of = {"a": [(256, 256)], "b": [(256, 256)], "c": [(1344, 768)]}
    sampled = {123456: "trajectory elements", 4096: "prompt length"}
    canvas, dependent, frames, real = classify(
        sorted(shapes), shapes, rasters_of, sampled, [clip_held]
    )
    assert canvas == [canvas_row], canvas
    # The order is the input's, so the test states membership rather than an ordering the
    # reader of the report never sees.
    assert set(dependent) == {sample_row, once_row}, dependent
    assert frames == [clip_missing], frames
    assert real == [twice_row], real

    # The same absence in one walk only: the canvas may explain it, and the tool says so
    # rather than guessing.
    single = dict(rasters_of, c=[(256, 256)])
    canvas, dependent, frames, real = classify([twice_row], shapes, single, {})
    assert canvas == [twice_row] and not real and not frames, (canvas, real, frames)

    # A row the file holds is not missing at all.
    canvas, dependent, frames, real = classify([], shapes, rasters_of, sampled)
    assert not canvas and not dependent and not real and not frames

    # `--wellformed` catches each of the three ways the file itself can be wrong: a header that
    # states a count it does not hold, a repeated line, and an operator `crates/ops` does not
    # define. Each is a failure of the file rather than a missing row, so no walk comparison sees
    # it - the count is quoted by every reader after it.
    # A walk with no rows must be refused rather than compared: every comparison
    # below passes when one side is empty, which is what this checks.
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        blank = root / "empty.txt"
        blank.write_text("")
        held = root / "held.txt"
        held.write_text(
            "# 1 data row\nrms_norm\tdtype_bf16_columns_5376_block_1024\n"
        )
        for compare in (by_shape, phantom):
            try:
                compare(
                    argparse.Namespace(walk=[str(blank)], fixture=str(held))
                )
            except SystemExit as refused:
                assert "a walk with no rows proves nothing" in str(refused), (
                    refused
                )
            else:
                raise AssertionError(
                    f"an empty walk was compared by {compare.__name__}"
                )

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        good = root / "good.txt"
        good.write_text(
            "# 2 data rows, 2 operators\n"
            "rms_norm\tdtype_bf16_columns_5376_block_1024\n"
            "gemm\tdtype_bf16_rows_64_columns_64\n"
        )
        assert wellformed(_Args(str(good))) == 0
        stale = root / "stale.txt"
        stale.write_text(
            "# 3 data rows\n"
            "rms_norm\tdtype_bf16_columns_5376_block_1024\n"
            "gemm\tdtype_bf16_rows_64_columns_64\n"
        )
        assert wellformed(_Args(str(stale))) == 1
        ops_wrong = root / "ops_wrong.txt"
        ops_wrong.write_text(
            "# 2 data rows, 3 operators\n"
            "rms_norm\tdtype_bf16_columns_5376_block_1024\n"
            "gemm\tdtype_bf16_rows_64_columns_64\n"
        )
        assert wellformed(_Args(str(ops_wrong))) == 1
        repeated = root / "repeated.txt"
        repeated.write_text(
            "# 3 data rows\nrms_norm\tone\nrms_norm\tone\ngemm\tother\n"
        )
        assert wellformed(_Args(str(repeated))) == 1
        unknown = root / "unknown.txt"
        unknown.write_text("# 1 data rows\nnot_an_op\tone\n")
        assert wellformed(_Args(str(unknown))) == 1

    # `--families` removes a field and the value after it, and nothing else.
    assert without("a_columns_128_block_1024", {"columns"}) == "a_block_1024"
    assert (
        without("a_columns_128_lines_4_block_1024", {"columns"})
        == "a_lines_4_block_1024"
    )
    assert without("a_block_1024", {"columns"}) == "a_block_1024"

    print(
        "ok   a canvas-class row absent from two canvases is real, from one it is the canvas,"
        " and the walk's own sample answers before either"
    )
    return 0


# Header lines that legitimately differ between a generated file and the checked-in
# copy: the generator names itself and the revision it ran at, and a row set is
# evidence only of the tree that produced it, so the copy carries that revision.
GENERATED_HEADER = re.compile(r"^# (generator|commit):", re.IGNORECASE)


def drift(
    generated: set[tuple[str, str]], fixture: set[tuple[str, str]]
) -> int:
    """Compare a fresh run of the generator with the checked-in row set."""
    only_generated = sorted(generated - fixture)
    only_fixture = sorted(fixture - generated)
    if not only_generated and not only_fixture:
        print(
            f"the list is the generator's rows: {len(fixture)} keys,"
            " no key missing, no key extra"
        )
        return 0
    for label, rows in (
        ("the generator states and the file does not", only_generated),
        ("the file holds and the generator does not", only_fixture),
    ):
        if rows:
            print(f"{len(rows)} key(s) {label}:")
            for operator, key in rows[:20]:
                print(f"  {operator}\t{key}")
    print(
        "a file that is not the generator's rows is a row set from another"
        " tree: re-cut it, or say which tree it came from - a store built"
        " from it would answer about a model that is not this one"
    )
    return 1


def generator_rows(root: str, extra: list[str]) -> set[tuple[str, str]]:
    """Run the model's own enumeration and return the keys it prints."""
    command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "minimax-h3",
        "--example",
        "minimax_h3_keys",
        "--",
        "--root",
        root,
        *extra,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        # The generator is another builder's tree: it compiles, it refuses, or it
        # panics. All three are answers about that tree, and a traceback here would
        # read as a broken instrument instead of the state of the tree.
        raise SystemExit(
            "the generator did not produce rows (exit "
            f"{result.returncode}):\n" + (result.stderr or result.stdout).strip()[-800:]
        )
    rows = set()
    for line in result.stdout.splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        operator, key = line.split("\t", 1)
        rows.add((operator, key))
    if not rows:
        raise SystemExit(
            "the generator printed no rows: an empty run is a build failure,"
            " not a row set"
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wellformed",
        action="store_true",
        help="the list alone: its stated count, its uniqueness, and its operator names",
    )
    parser.add_argument(
        "--phantom",
        action="store_true",
        help="operators the list holds that no walk in the set launches",
    )
    parser.add_argument(
        "--families",
        action="store_true",
        help="per operator: distinct keys launched, and distinct keys with `--fields` removed",
    )
    parser.add_argument(
        "--fields",
        default="columns",
        help="the key fields `--families` removes, comma-separated",
    )
    parser.add_argument(
        "--walk",
        required=False,
        action="append",
        help="what the forward reported, one launch a line; repeat it to take a union of stages",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="prove the classification on walks and a file it makes itself",
    )
    parser.add_argument(
        "--by-shape",
        action="store_true",
        help="compare with the element type normalised, so a walk of one deployment checks the "
        "shapes a file serving several deployments must hold; fails on a shape absent at every type",
    )
    parser.add_argument(
        "--fixture", help="the store list the caller builds from"
    )
    parser.add_argument(
        "--generator",
        metavar="ROOT",
        help="compare the checked-in list with a fresh run of the model's own"
        " enumeration at ROOT",
    )
    parser.add_argument(
        "--generator-flag",
        action="append",
        default=[],
        help="a flag passed through to the generator, such as `--types`;"
        " repeatable",
    )
    parser.add_argument(
        "--unbound",
        action="store_true",
        help="list every fixture line no launch bound (the normal case for one deployment, so the "
        "default prints the count and a few)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    # `--families` reads launches only and `--wellformed` reads the list only: neither has a
    # second side to be given.
    if args.wellformed and not args.fixture:
        parser.error("--wellformed reads the list: give it --fixture")
    if not args.wellformed and not args.generator and (
        not args.walk or (not args.fixture and not args.families)
    ):
        parser.error(
            "--walk and --fixture name the two sides, or --families reads the walks"
            " alone, or --self-test proves the tool"
        )
    if args.generator:
        if not args.fixture:
            parser.error(
                "--generator compares with the list: give it --fixture"
            )
        fixture = read_lines(args.fixture)
        generated = generator_rows(args.generator, args.generator_flag)
        return drift(generated, set(fixture))
    if args.by_shape:
        return by_shape(args)
    if args.families:
        return families(args)
    if args.phantom:
        return phantom(args)
    if args.wellformed:
        return wellformed(args)
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
            f"rows the list holds and no walk in this set binds ({len(unbound)}):"
        )
        shown = unbound if args.unbound else unbound[:8]
        # A row no walk in this set binds is not excused by the set being narrow: it is
        # either a construction's row (name the construction that launches it) or a cell
        # nothing asks for. The walk prints which axes it bound, so a reader can tell
        # whether the set covers the deployment.
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
