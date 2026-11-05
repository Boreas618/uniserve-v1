#!/bin/sh
# Re-cut the two H3 deployment fixtures from the walk, and certify both.
#
# A fixture is one deployment's rows: the dense walk and the sparse one, each at the deployment's own
# bounds and each component's own element type. Both are produced here from `minimax-h3`'s own
# enumeration, so a fixture cannot disagree with the walk about what the model launches — the failure
# this lane has hit most often.
#
# Usage:
#   tools/recut_h3_fixtures.sh <checkpoint root> <element-types json> [<commit>]
#
#   <checkpoint root>     e.g. /opt/uv/models/MiniMax-H3
#   <element-types json>  the `--types` fallback, used where the host has no shard headers,
#                         e.g. artifacts/h3-reference/component-tensors.json
#   <commit>              the revision the rows are stamped with; defaults to HEAD
#
# Everything is written to a staging directory first and only moved into `notes/programs/` once both
# files certify, so an interrupted run leaves the checked-in fixtures untouched rather than half
# replaced.

set -eu

root=${1:?usage: recut_h3_fixtures.sh <checkpoint root> <element-types json> [<commit>]}
types=${2:?usage: recut_h3_fixtures.sh <checkpoint root> <element-types json> [<commit>]}
# The revision the rows are stamped with. In a git tree it defaults to HEAD and the tree must be clean:
# a walk reads the working tree, so a fixture cut from a dirty tree would be stamped with a commit that
# did not produce it, and the stamp is the only thing tying a row set to the code that enumerated it.
# In an exported tree there is no git, so the revision must be given.
if head=$(git rev-parse --short HEAD 2>/dev/null); then
    commit=${3:-$head}
    if [ -n "$(git status --porcelain -- crates/ 2>/dev/null)" ]; then
        echo 'the working tree has uncommitted changes under crates/, so no commit describes this walk:'
        git status --porcelain -- crates/
        echo 'commit or stash them, then run this again'
        exit 1
    fi
else
    commit=${3:?this tree has no git, so pass the revision the walk is stamped with as the third argument}
fi

# The deployment arguments. All of these are part of the fixture's identity and the walk prints them
# back into its own header, so a fixture states the deployment it belongs to without anyone retyping it.
# The clip length is a count of **generated frames**, and a generated frame count is `17n + 5`: the
# legal clips are 22, 39, 56, ... A round number like 256 is not one of them, so a walk cut at 256
# states rows for a clip no request can ask for.
samples=22
# The six canvases the model serves, in the walk's own `HxW` order. **The authority is
# `crates/models/minimax-h3/src/config/canvas.rs`'s `NAMED_CANVASES`**, and this line is a second
# statement of it — `programs --rasters offered` reading `offered_canvases()` is owed so that a
# canvas added to the model reaches every fixture without anyone editing this file. Until then, a
# raster that is not in the model must not appear here: 480x832 was in this line for days and every
# fixture it produced named a canvas no request asks for.
rasters=672x1536,768x1344,768x1024,768x768,1024x768,1344x768
tokens=4096
conditioned=true
elements=123456
# The track is a **function of the clip**, not a second number that can disagree with it:
# `packing::audio_samples(frames) = audio_latent_frames(frames) * AUDIO_HOP`, and for 22 frames that
# is 29600. The pair is part of the fixture's identity because the audio VAE's `scaled_add` widths
# are derived from it: measured, 256/44100 and 22/29600 state the same 150 rows with **12 `scaled_add`
# rows different** and nothing else, so a fixture cut at one pair does not serve a request at the
# other. The walk still takes `--track` as a free number; deriving it from `--samples` there is owed,
# exactly as `rows_bound()` is owed, and until then this is the pair both enumerations accept — the
# recorder refuses any other by name.
track=29600

# The sparse deployment's attention: 64 tiles of 0.9 budget, keyed at the deployment's own bound's next
# power of two. The bound is the deployment's, not one request's, and the walk refuses any other class
# by name.
sparse=64,0.9,8192

# The deployment's packed bound, restated here only because the walk still takes `--rows` as a flag.
# **This is the value `rows_bound()` computes**: a bound the model can exceed belongs in the
# descriptor with the launch refusing outside it, not in a walk argument that works because nothing
# tested past it. Delete these two lines the moment the walk reads `rows_bound()`; until then a walk
# cut below them stamps rows for a packed count the deployment cannot be asked to be.
# These are the fallback, not the answer: the probe below runs the walk with no `--rows` and uses
# them only when the walk refuses it. `rows_bound()` is the one authority, and h3-builder04 measured
# the walk's bound (303056) apart from the serving path's (304912/304976) — so a constant here is a
# second statement of a live disagreement. **They leave this file when the walk reads the bound.**
rows_dense=306896
rows_sparse=353152
rows_flag=()
if ! RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" --types "$types" \
    --samples "$samples" --rasters 768x768 --tokens "$tokens" --tp 1 --conditioned false \
    --elements "$elements" --track "$track" >/dev/null 2>&1; then
    rows_flag=(--rows)
fi

staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

walk_dense="$staging/walk-dense.txt"
walk_sparse="$staging/walk-sparse.txt"
fixture_dense="$staging/fixture-dense.txt"
fixture_sparse="$staging/fixture-sparse.txt"

run() {
    name=$1
    shift
    printf '  %-28s' "$name"
    if "$@" >/dev/null 2>&1; then
        echo 'ok'
    else
        echo 'FAILED'
        "$@" 2>&1 | tail -5
        exit 1
    fi
}

echo "re-cutting H3 fixtures at $commit"

# The walk this tool drives is the only place a deployment's rows are stated, and the properties it
# needs have been absent from the tips this tool has been run against more than once: each component's
# own element type (`--types`), the offered canvases, the deployment's own packed bound (so no
# `--rows`), and no `--dtype`. A walk without `--types` keys every row from one element type, so a
# fixture cut from it describes one type where the model runs two - and no delta can see that, because
# the fixture and the walk agree with each other. Diagnose it by name rather than letting the walk's
# own flag error read as a tool fault.
probe=$(mktemp)
if ! RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" --types "$types" \
    --samples "$samples" --rasters 768x768 --tokens "$tokens" --tp 1 --conditioned false \
    --elements "$elements" --track "$track" >"$probe" 2>&1; then
    if grep -q 'unknown flag --types' "$probe"; then
        cat >&2 <<'EOF'
the walk at this tip takes no `--types`, so it cannot state each component's own element type:
  the walk is missing the deployment properties (--types/deployment_dtype, offered_canvases,
  rows_bound, and no --dtype), and a fixture cut here would key every row from one element type.
  Land the walk's element types first, then re-cut: the recorder keys its rows per component, so the
  two enumerations cannot be compared until both do.
EOF
        rm -f "$probe"
        exit 1
    fi
fi
rm -f "$probe"

# 1. The walks. No `--dtype`: each component's own type, read from the checkpoint (or from `--types`
#    where this host has no shards). `--rows` is passed only because the walk still takes it as a flag;
#    the two values are what `rows_bound()` computes.
echo 'walk:'
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" \
    --samples "$samples" --rasters "$rasters" --tokens "$tokens" ${rows_flag:+"${rows_flag[@]}" "$rows_dense"} \
    --tp 1 --conditioned "$conditioned" --elements "$elements" --track "$track" \
    --types "$types" >"$walk_dense"
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" \
    --samples "$samples" --rasters "$rasters" --tokens "$tokens" \
    --tp 1 --conditioned "$conditioned" --elements "$elements" --track "$track" \
    --types "$types" ${rows_flag:+"${rows_flag[@]}" "$rows_sparse"} --sparse "$sparse" >"$walk_sparse"
printf '  dense %s rows, sparse %s rows\n' \
    "$(grep -c '^[a-z]' "$walk_dense")" "$(grep -c '^[a-z]' "$walk_sparse")"

# 2. The recorder, against the same walk print. **This is the only check that can see a walk which
#    forgets a launch.** `keys_delta` compares the fixture with its walk and `key_coverage` asks
#    whether a package declares a key; both are green on a walk that states 139 rows while the model
#    launches 175, because the walk is what they take as given. The recorder asks the model instead:
#    every launch it records must be a row the walk states, or the fixture describes a model that does
#    not exist. A non-zero count is a walk defect, not a tool fault, so the rows are printed.
echo 'recorder:'
recorder() {
    name=$1
    walk=$2
    shift 2
    if ! RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p minimax-h3 --example record -- --component all \
        --conditioned "$conditioned" --tokens "$tokens" --elements "$elements" \
        --track "$track" --extents "$extents" --types "$types" --against "$walk" \
        "$@" >"$staging/record-$name.txt" 2>&1; then
        printf '  %-8sFAILED\n' "$name"
        tail -5 "$staging/record-$name.txt"
        recorder_failed=1
        return
    fi
    line=$(grep -E '^# [0-9]+ recorded' "$staging/record-$name.txt" || true)
    if [ -z "$line" ]; then
        printf '  %-8sthe recorder printed no summary, which reads exactly like a pass\n' "$name"
        tail -5 "$staging/record-$name.txt"
        recorder_failed=1
        return
    fi
    unstated=$(printf '%s' "$line" | sed -n 's/.*\([0-9][0-9]*\) only recorded.*/\1/p')
    printf '  %-8s%s\n' "$name" "$line"
    if [ -z "$unstated" ] || [ "$unstated" != "0" ]; then
        echo "the model launches rows this walk does not state (${unstated:-unparsed}):" >&2
        grep '^recorded, not stated:' "$staging/record-$name.txt" >&2 || true
        echo "a fixture cut here is short of launches the model makes; fix the walk, not the fixture" >&2
        recorder_failed=1
    fi
}

# `--extents` is the recorder's own fixture of tensor shapes; it is committed beside the fixtures.
extents=${extents:-notes/programs/h3-extents.json}
# **Both deployments report before the tool fails.** A gate that exits at the first one states a
# count for one deployment and reads as if it had spoken about the other: with dense short the tool
# stopped there, so no run ever printed the sparse count, and a reader comparing the two fixtures had
# one number where two exist.
recorder_failed=0
recorder dense "$walk_dense"
recorder sparse "$walk_sparse" --sparse 64,0.9
if [ "$recorder_failed" != "0" ]; then
    echo "the re-cut stops: a walk above is short of launches the model makes, so no fixture is written" >&2
    exit 1
fi

# 3. The blocked block, pasted from the tool rather than written by hand. A stale exemption is as wrong
#    as a missing one and harder to see, because a test comparing failures to the block goes green on a
#    file that exempts what nothing fails on.
echo 'declared blocked:'
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p program-server --example key_coverage -- \
    --fixture "$walk_dense" --target sm_100a --declared-blocked \
    2>/dev/null | grep '^# blocked:' >"$staging/blocked-dense.txt" || true
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p program-server --example key_coverage -- \
    --fixture "$walk_sparse" --target sm_100a --declared-blocked \
    2>/dev/null | grep '^# blocked:' >"$staging/blocked-sparse.txt" || true
printf '  dense %s, sparse %s\n' \
    "$(wc -l <"$staging/blocked-dense.txt")" "$(wc -l <"$staging/blocked-sparse.txt")"

# 4. The fixtures: the header, the block and the rows, from the walk's own print.
echo 'fixtures:'
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example h3_fixture -- \
    --walk "$walk_dense" --commit "$commit" --blocked "$staging/blocked-dense.txt" >"$fixture_dense"
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example h3_fixture -- \
    --walk "$walk_sparse" --commit "$commit" --blocked "$staging/blocked-sparse.txt" >"$fixture_sparse"

# 5. Certify: the fixture must be the walk's rows, exactly, and every key must be covered or declared
#    blocked. A fixture that fails either is one a store must not be built from.
echo 'certify:'
status=0
for pair in "dense:$fixture_dense:$walk_dense" "sparse:$fixture_sparse:$walk_sparse"; do
    name=${pair%%:*}
    rest=${pair#*:}
    fixture=${rest%%:*}
    walk=${rest#*:}

    if RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p ops --example keys_delta -- \
        --fixture "$fixture" --walk "$walk" 2>&1 | grep -q 'the fixture is the walk.s rows'; then
        printf '  %-10s delta   ok\n' "$name"
    else
        printf '  %-10s delta   FAILED — the fixture is not the walk\n' "$name"
        status=1
    fi

    coverage=$(RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p program-server --example key_coverage -- \
        --fixture "$fixture" --target sm_100a 2>&1 | grep -E 'not covered|every key' | head -1)
    # A coverage check that prints nothing is a broken check, not a clean fixture: the tool once
    # wrote both fixtures while this line was empty, and an empty line reads exactly like a pass.
    if [ -z "$coverage" ]; then
        echo "  $name coverage printed nothing — refusing to certify a fixture whose coverage is unknown" >&2
        exit 1
    fi
    printf '  %-10s coverage %s\n' "$name" "$coverage"
    # A key the fixture names, that no package declares, and that the blocked block does not excuse,
    # is the one state a store must not be built from. The blocked block is pasted from
    # `--declared-blocked` above, so an excused key never reaches this test.
    case "$coverage" in
        *" 0 of "*) ;;
        *)
            if [ ! -s "$staging/missing-$name.txt" ]; then
                echo "  $name has uncovered keys and no block to excuse them" >&2
                exit 1
            fi
            ;;
    esac
done

if [ "$status" -ne 0 ]; then
    echo 'a fixture failed certification; notes/programs/ was not touched'
    exit 1
fi

# 5. Move both into place. Only now, and both together, so the two fixtures in the tree are always from
#    one walk at one commit.
cp "$fixture_dense" notes/programs/minimax-h3-keys.txt
cp "$fixture_sparse" notes/programs/minimax-h3-sparse-keys.txt
echo 'written: notes/programs/minimax-h3-keys.txt, notes/programs/minimax-h3-sparse-keys.txt'
