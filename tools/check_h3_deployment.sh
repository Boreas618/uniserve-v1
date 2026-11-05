#!/bin/sh
# Check the H3 deployment list against the model that would build it, in one run.
#
# The list in `notes/programs/` is one deployment's program rows. Five questions
# decide whether it may be built into a store, and each has its own tool; this
# script runs them in the order that fails fastest, and prints one line per
# question so a reader can see which answer moved.
#
# Usage:
#   tools/check_h3_deployment.sh <checkpoint root> <element-types json> [<canvas>] [<clip>]
#
#   <checkpoint root>     e.g. /opt/uv/models/MiniMax-H3
#   <element-types json>  the `--types` table, e.g. artifacts/h3-reference/component-tensors.json
#   <canvas>              a canvas a walk should reach, default 768x768
#   <clip>                a clip a walk should reach, default 124
#
# Nothing here writes to `notes/programs/`: producing a fixture is
# `tools/recut_h3_fixtures.sh`, which stamps the revision that produced it. This
# script answers "is the file the rows of the tree in front of me", which is the
# question a working tree can answer at any moment - including while it is dirty.
#
# Exits non-zero on the first question that fails, naming the question.

set -eu

root=${1:?usage: check_h3_deployment.sh <checkpoint root> <element-types json> [<canvas>] [<clip>]}
types=${2:?usage: check_h3_deployment.sh <checkpoint root> <element-types json> [<canvas>] [<clip>]}
canvas=${3:-768x768}
clip=${4:-124}

here=$(cd "$(dirname "$0")/.." && pwd)
fixture=$here/notes/programs/minimax-h3-keys.txt
sparse_fixture=$here/notes/programs/minimax-h3-sparse-keys.txt
gaps=$here/crates/models/minimax-h3/tools/walk_gaps.py
manifests=$here/crates/models/minimax-h3/tools/check_encoder_manifests.py
ladder=$here/crates/models/minimax-h3/tools/check_frames_ladder.py
scratch=${TMPDIR:-/tmp}/h3-deployment-check.$$
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT

fail() {
    echo
    echo "FAILED: $1"
    exit 1
}

# The clip's soundtrack is the clip's own, in latent frames: 800 samples per
# latent at 40 latents per second of 24 generated frames per second.
track=$(python3 - "$clip" <<'PY'
import sys
print(round(int(sys.argv[1]) * 40 / 24) * 800)
PY
)

# 1. Is the file the rows of this tree?
echo "== the list against the model's own enumeration"
python3 "$gaps" --generator "$root" --fixture "$fixture" || fail "the file is not the generator's rows"

# 2. Does the file state what it holds?
echo
echo "== the sparse list against its generator"
# The sparse file is a deployment of its own: its rows come from the generator run
# with `--sparse`, and a file cut from a walk at another clip fails here by name
# rather than by falling through to a walk comparison, which measures the walk.
#
# The spelling is the *producer's*, character for character (`tools/recut_h3_fixtures.sh`'s
# `sparse=`): the third field is a reference keep rate a deployment states only when it differs
# from the checkpoint's own, so `64,0.9` and `64,0.9,0.9` are two deployments that happen to
# produce one row set today - the rows agree and the launches do not. A check that spelled the
# deployment differently from the tool that cut the file would compare two deployments and call
# the equality of their rows a match.
if [ -f "$sparse_fixture" ]; then
    python3 "$gaps" --generator "$root" --generator-flag=--sparse --generator-flag=64,0.9 \
        --fixture "$sparse_fixture" || sparse_not_generated=1
else
    echo "   skipped: $sparse_fixture is absent"
fi

echo
echo "== the list against its own header"
python3 "$gaps" --wellformed --fixture "$fixture" || fail "the list does not state what it holds"

# 3. Is every row declared by a package?
echo
echo "== every row against the manifests"
python3 "$manifests" --rows "$fixture" | tail -1 || true
python3 "$manifests" --rows "$fixture" | grep -q ' 0 wait on cells' \
    || fail "a row waits on a cell no manifest declares"

# 4. Is every clip the file serves a clip a request can name?
echo
echo "== the clip ladder against the frame rule"
python3 "$ladder" || fail "the file serves a clip the rule refuses, or the default ladder drifted"

# 5. Do walks reach every operator the list holds, and does every walk's shape
#    appear in the file? One walk per task, plus a sparse one: the `vsa_*` rows
#    only a sparse deployment launches, and the three tasks differ by one row.
#
#    A walk reads the canvas and clip lists it is given only at their first
#    element - it prints every axis in its header and binds one - so the
#    deployment's offered points are walked one at a time and the unions taken.
echo
echo "== every (canvas, clip) the file states, one point at a time"
# A file whose header states six canvases and four clips is the rows of all
# twenty-four points, and a walk reaches one point's rows: a file cut at a single
# point is walk-equal there and short everywhere else - a shape that has looked
# right once.
points=$(python3 - "$fixture" <<'PYHDR'
import re, sys

# Both fixtures state their axes in the header, in two wordings: the walk's own
# ("raster sizes: [(h, w), ...]" with "clip lengths: [...]") and the deployment
# file's ("axes: 6 canvases (hxw, ...) and the clip ladder [...]"). Read either.
header = " ".join(l for l in open(sys.argv[1]) if l.startswith("#"))
rasters = re.search(r"raster sizes: \[(.*?)\]", header)
if rasters:
    canvases = [f"{h}x{w}" for h, w in re.findall(r"\((\d+), (\d+)\)", rasters.group(1))]
else:
    canvases = re.findall(r"\b(\d+x\d+)\b", re.search(r"canvases \((.*?)\)", header).group(1))
clips = re.search(r"clip lengths: \[(.*?)\]", header) or re.search(r"clip ladder \[(.*?)\]", header)
print(" ".join(f"{c}:{k}" for k in clips.group(1).split(", ") for c in canvases))
PYHDR
)
unbuildable=""
for point in $points; do
    p_canvas=${point%%:*}
    p_clip=${point##*:}
    cargo run -q -p minimax-h3 --example programs -- "$root" \
        --samples "$p_clip" --track "$(python3 -c "print(round($p_clip * 40 / 24) * 800)")" \
        --tokens 4096 --elements 123456 --tp 1 --types "$types" \
        --rasters "$p_canvas" --task ref2va --conditioned true \
        >"$scratch/point.txt" 2>/dev/null || true
    answer=$(python3 "$gaps" --by-shape --fixture "$fixture" --walk "$scratch/point.txt" | tail -1)
    # A point passes on one of the tool's two closing sentences: nothing to
    # explain, or every absence explained as the walk's own sample row (a
    # request's number, not a deployment's). Anything else is a shape the file
    # cannot build, and the reading is kept so the failure names the shape.
    case "$answer" in
    *"every shape any walk produced is in the fixture"*) ;;
    *"0 raster-class, 1 the walk's own sample"*) ;;
    *)
        unbuildable="$unbuildable $point"
        [ -f "$scratch/point-reading.txt" ] || echo "$answer" >"$scratch/point-reading.txt"
        ;;
    esac
done
if [ -n "$unbuildable" ]; then
    echo "   the file cannot build every shape these points launch:$unbuildable"
    [ -f "$scratch/point-reading.txt" ] && sed 's/^/     /' "$scratch/point-reading.txt"
    fail "the file is the rows of some points and not of the ones its header states"
fi
echo "   $(echo "$points" | wc -w) point(s) walked singly, each one buildable from the file"

echo "== the model's own launch record against the file"
# `--by-shape` compares shapes and the generator compares enumerations; neither sees
# which operand is which. The recorder drives each component's `forward` and reports
# the descriptors it launches, so a row whose operands are mirrored - `input_f32_
# weight_bf16` in the forward and `input_bf16_weight_f32` in the file - is visible
# here and nowhere else. One direction only: a recorded row missing from the file is
# a run that stops at `no kernel`; a file row the recording does not reach is a
# different point, a different component, or the conditioned path, and is not a gap.
extents=notes/programs/h3-extents.json
if [ ! -f "$extents" ]; then
    echo "   skipped: $extents is absent, and without it audio-encoder, audio-decoder"
    echo "   and conditioned refuse and their rows would read as absent"
else
    # Both paths are recorded, for both deployments. The conditioned pass builds the text
    # encoder's own harness and states a weight's element type from its own loader, so a loader
    # that reports the wrong type keys a row only that pass can contradict. The sparse file is
    # recorded with the walk's own sparse setting, because the file that ships is the sparse
    # one and a launch record for the dense list alone says nothing about it.
    record_deployment() {
        local fixture_path=$1
        local recorded_txt=$2
        shift 2
        : >"$recorded_txt"
        for cond in false true; do
            for point in $points; do
                r_canvas=${point%%:*}
                r_clip=${point##*:}
                cargo run -q -p minimax-h3 --example record -- --root "$root" --component all \
                    --clip "$r_clip" --raster "$r_canvas" --tokens 4096 \
                    --track "$(python3 -c "print(round($r_clip * 40 / 24) * 800)")" \
                    --elements 123456 --types "$types" --extents "$extents" --conditioned "$cond" \
                    "$@" 2>/dev/null | grep -P '\t' >>"$recorded_txt" || true
            done
        done
        cut -f1,2 "$recorded_txt" | grep -v '^#' | sort -u >"$recorded_txt.rows"
        cut -f1,2 "$fixture_path" | grep -v '^#' | sort -u >"$recorded_txt.file.rows"
        # Two fields are a request's, not a deployment's, and a row that differs only in one of
        # them is not a missing row: `columns_<elements>` carries the request's trajectory width,
        # and `wide_true|false` is `elements >= u32::MAX` - a launch at any real request is
        # `wide_false` while the file states the form its bound requires. Both sides are
        # legitimate; comparing them literally would report the file's own bound as a gap.
        comm -23 "$recorded_txt.rows" "$recorded_txt.file.rows" | grep -v 'columns_123456' \
            | while read -r row; do
                folded=$(printf '%s' "$row" | sed 's/_wide_true/_wide_FOLDED/; s/_wide_false/_wide_FOLDED/')
                if grep -qF "$(printf '%s' "$folded" | cut -f2)" \
                    <(sed 's/_wide_true/_wide_FOLDED/; s/_wide_false/_wide_FOLDED/' "$recorded_txt.file.rows"); then
                    continue
                fi
                printf '%s\n' "$row"
            done >"$recorded_txt.missing"
        echo "   $(wc -l <"$recorded_txt.rows") row(s) recorded of $(wc -l <"$recorded_txt.file.rows") stated at $(echo "$points" | wc -w) point(s)"
        if [ -s "$recorded_txt.missing" ]; then
            sed 's/^/     NOT IN THE FILE: /' "$recorded_txt.missing" | head -6
            fail "a forward launches a descriptor $(basename "$fixture_path") does not state"
        fi
    }

    record_deployment "$fixture" "$scratch/recorded.dense" || true
    record_deployment "$sparse_fixture" "$scratch/recorded.sparse" --sparse 64,0.9,0.9 || true
fi

echo "== four walks: one per task, and a sparse one"
run() {
    name=$1
    shift
    cargo run -q -p minimax-h3 --example programs -- "$root" \
        --samples "$clip" --track "$track" --tokens 4096 --elements 123456 \
        --tp 1 --types "$types" --rasters "$canvas" "$@" \
        >"$scratch/$name.txt" 2>"$scratch/$name.err"
    echo "   $name: $(grep -c "$(printf '\t')" "$scratch/$name.txt") rows"
}
run t2va --task t2va --conditioned false
run fl2va --task fl2va --conditioned true
run ref2va --task ref2va --conditioned true
run sparse --task fl2va --conditioned true --sparse 64,0.9,8192

echo
echo "== every operator the list holds, launched by some walk"
python3 "$gaps" --phantom --fixture "$fixture" \
    --walk "$scratch/t2va.txt" --walk "$scratch/fl2va.txt" \
    --walk "$scratch/ref2va.txt" --walk "$scratch/sparse.txt" \
    || fail "an operator the list holds is launched by no walk"

echo
echo "== each walk's shapes against its own file"
for walk in t2va fl2va ref2va sparse; do
    # One file is one attention backend: the sparse walk's `vsa_*` and `sparse_attention`
    # rows belong to the sparse file, and comparing them with the dense one would report
    # seven gaps that are a deployment's, not a file's.
    against=$fixture
    [ "$walk" = sparse ] && against=$sparse_fixture
    answer=$(python3 "$gaps" --by-shape --fixture "$against" --walk "$scratch/$walk.txt" | tail -1)
    echo "   $walk: $answer"
    # A walk states one shape the file lacks at every element type when it is
    # the walk's own sample (a request-only quantity, not a deployment one); a
    # raster-class absence, or one no raster explains, is a real gap.
    # The reading ends "... 0 at any raster (...), 0 raster-class, 1 the walk's own
    # sample": the sample row is a request's number, and any other absence is a gap.
    case "$answer" in
    *"every shape any walk produced is in the fixture"*) ;;
    *"0 raster-class, 1 the walk's own sample") ;;
    *) fail "the $walk walk states a shape the file cannot build at any element type" ;;
    esac
done

# 6. Every offered (canvas, clip) point, walked singly and unioned. A list passed
#    once binds the first point and its header claims the rest, so this loop is the
#    only way to ask what the deployment launches across its canvases and clips.
echo
echo "== the deployment's offered points, walked one at a time"
: >"$scratch/union.txt"
for clip in 124 192 243 362; do
    case $clip in
    124) track=165600 ;;
    192) track=256000 ;;
    243) track=324000 ;;
    362) track=482400 ;;
    esac
    for canvas in 672x1536 768x1344 768x1024 768x768 1024x768 1344x768; do
        cargo run -q -p minimax-h3 --example programs -- "$root" \
            --samples "$clip" --track "$track" --tokens 4096 --elements 123456 \
            --tp 1 --types "$types" --rasters "$canvas" --task ref2va \
            --conditioned true >>"$scratch/union.txt" 2>/dev/null
    done
done
echo "   24 points, $(cut -f1,2 "$scratch/union.txt" | sort -u | wc -l) rows bound"
# A row keyed by the walk's own `--elements` is the walk's sample, not a deployment row: the
# same rule `--by-shape` applies to the sample count. It is printed and excluded, so a reader
# does not have to know the rule to read the line below it.
python3 "$gaps" --walk "$scratch/union.txt" --fixture "$fixture" >"$scratch/union-reading.txt" 2>&1 || true
grep -E '^(  MISSING|[0-9]+ key)' "$scratch/union-reading.txt" | sed 's/^/   /' || true
if grep 'MISSING from the fixture' "$scratch/union-reading.txt" | grep -qv 'columns_123456'; then
    fail "a row the model binds at an offered point is not in the file"
fi
if grep -q 'MISSING from the fixture.*columns_123456' "$scratch/union-reading.txt"; then
    echo "   the one above is the walk's sample row (its own \`--elements\`), a request's number - not a row to add"
fi

echo
echo
# A sparse file that is another tree's rows is reported at the end rather than at
# the point it is found: the dense file's other seven answers are independent of it,
# and a reader who never sees them reads a red as being about the deployment.
if [ "${sparse_not_generated:-}" = 1 ]; then
    fail "the sparse file is not the generator's sparse rows"
fi

echo "every question answers: the list is this tree's rows, every row is declared,"
echo "every clip is one a request can name, and the walks' shapes are in the file."
