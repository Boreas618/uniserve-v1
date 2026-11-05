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
fixture=$here/notes/programs/minimax-h3-deployment-keys.txt
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
echo "== each walk's shapes against the file"
for walk in t2va fl2va ref2va sparse; do
    answer=$(python3 "$gaps" --by-shape --fixture "$fixture" --walk "$scratch/$walk.txt" | tail -1)
    echo "   $walk: $answer"
    # A walk states one shape the file lacks at every element type when it is
    # the walk's own sample (a request-only quantity, not a deployment one); a
    # raster-class absence, or one no raster explains, is a real gap.
    case $answer in
    *" 0 at any raster, 0 raster-class, 1 the walk's own sample") ;;
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
python3 "$gaps" --walk "$scratch/union.txt" --fixture "$fixture" \
    | grep -E '^(  MISSING|[0-9]+ key)' || true
if python3 "$gaps" --walk "$scratch/union.txt" --fixture "$fixture" \
    | grep -q 'MISSING from the fixture'; then
    fail "a row the model binds at an offered point is not in the file"
fi

echo
echo "every question answers: the list is this tree's rows, every row is declared,"
echo "every clip is one a request can name, and the walks' shapes are in the file."
