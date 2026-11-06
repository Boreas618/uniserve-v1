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
# `SAMPLES` and `TRACK` are the deployment's clip lengths and their audio tracks, and they come from
# outside because the deployment states them: `124,192,243,362` are the delivery's durations snapped to
# the `17n + 5` rung. A clip set written here is a second statement of the deployment, which is the
# defect this file has already been fixed for twice (a raster and a bound).
samples=${SAMPLES:-260}
# **No canvas list lives here.** The walk is the model's own generator, which reads the deployment's
# canvases from `crates/models/minimax-h3/src/config/canvas.rs`'s `NAMED_CANVASES`; a list typed in
# this file would be a second statement of them, and the one this line used to carry named 480x832 -
# a canvas no request asks for - for days, which every fixture it produced inherited. The probes below
# take one literal canvas because they ask a yes/no question (`--rows` refused, `--types` accepted),
# never a row set.
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
# The walk derives the track from the clip (`--samples [260] imply tracks [346400]`) and refuses a
# mismatched pair, so this file passes no track to it. `TRACK` is still what the *recorder* takes.
track=${TRACK:-346400}
# The task is required and its rows differ: `t2va` states one row the conditioned walks do not.
task=${TASK:-ref2va}

# The sparse deployment's attention: 64 tiles of 0.9 budget, keyed at the deployment's own bound's next
# power of two. The bound is the deployment's, not one request's, and the walk refuses any other class
# by name.
# The deployment's sparse attention, in the one grammar both halves take:
# `tile_rows,sparsity[,reference_keep]`. The tile **bound** is not an argument - the layout derives
# it (351 232 packed rows at this deployment), and a third field here was a second statement of it
# that only one of the two parsers accepted.
sparse=64,0.9

# The deployment's packed bound, restated here only because the walk still takes `--rows` as a flag.
# **This is the value `rows_bound()` computes**: a bound the model can exceed belongs in the
# descriptor with the launch refusing outside it, not in a walk argument that works because nothing
# tested past it. Delete these two lines the moment the walk reads `rows_bound()`; until then a walk
# cut below them stamps rows for a packed count the deployment cannot be asked to be.
# These are the fallback, not the answer: the probe below runs the walk with no `--rows` and uses
# them only when the walk refuses it. `rows_bound()` is the one authority, and h3-builder04 measured
# the walk's bound (303056) apart from the serving path's (304912/304976) — so a constant here is a
# second statement of a live disagreement. **They leave this file when the walk reads the bound.**
# No fallback bound. The walk derives it from the deployment (`rows_bound`), and a
# number typed here describes whatever deployment was current when it was typed -
# 306896 was the 256-clip era. A tip that demands `--rows` is refused below rather
# than handed a constant.
# The walk derived its track from the clip on one tip and required it on another, so this file
# probes: it passes `--track` only when the walk refuses its absence. A file that states one
# deployment's convention breaks on the other, which is how `--rows` and `--samples` both went wrong.
track_flag=()
if ! RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" --types "$types" \
    --task "$task" --samples "$samples" --rasters 768x768 --tokens "$tokens" --tp 1 \
    --conditioned "$conditioned" --elements "$elements" >/dev/null 2>&1; then
    track_flag=(--track)
fi
rows_flag=()
if ! RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" --types "$types" \
    --samples "$samples" --rasters 768x768 --tokens "$tokens" --tp 1 --conditioned "$conditioned" \
    --elements "$elements" ${track_flag:+"${track_flag[@]}" "$track"} --task "$task" >/dev/null 2>&1; then
    rows_flag=(--rows)
fi
if [ ${#rows_flag[@]} -ne 0 ]; then
    echo "the re-cut stops: this walk demands --rows, and the bound is the deployment's." >&2
    echo "  A tool that types one describes a deployment it cannot be asked to be." >&2
    exit 1
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
    --samples "$samples" --rasters 768x768 --tokens "$tokens" --tp 1 --conditioned "$conditioned" \
    --elements "$elements" ${track_flag:+"${track_flag[@]}" "$track"} --task "$task" >"$probe" 2>&1; then
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

# 1. The walks, from the **generator** and not from the `programs` example.
#
#    A fixture is one deployment's rows, and a deployment is a canvas set, a clip ladder and a task.
#    The `programs` example takes those as one request's arguments - `--track` is a single number, so
#    it cannot state a clip ladder at all (`--samples 124,192,243,362 --track 346400` is refused by
#    name). Every fixture this tool cut before this line was changed was therefore a one-clip cut
#    wearing a deployment's header, and no comparison could see it, because both sides were row sets.
#
#    `minimax_h3_keys` is the model's own enumeration: it reads the checkpoint, derives the canvases
#    and the clip ladder from the deployment, and takes `--sparse` for the video-sparse deployment.
#    The request arguments above stay on the recorder step, where they describe the one request it
#    drives - which is what they are for.
echo 'walk:'
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example minimax_h3_keys -- --root "$root" --commit "$commit" \
    >"$walk_dense"
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example minimax_h3_keys -- --root "$root" --sparse "$sparse" \
    --commit "$commit" >"$walk_sparse"
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
        withheld="$name"
    fi
}

# `--extents` is the recorder's own fixture of tensor shapes; it is committed beside the fixtures.
extents=${extents:-notes/programs/h3-extents.json}
# **Both deployments report before the tool fails.** A gate that exits at the first one states a
# count for one deployment and reads as if it had spoken about the other: with dense short the tool
# stopped there, so no run ever printed the sparse count, and a reader comparing the two fixtures had
# one number where two exist.
# A short walk withholds **its own** deployment and nothing else. The two fixtures are two
# deployments' row sets at one commit; a row the sparse deployment launches and the sparse walk lacks
# is no reason to withhold the dense file, which is certified separately below. The run's exit code
# still reports the shortfall, and a reader gets both counts whatever happens.
recorder_failed=0
withheld=
recorder dense "$walk_dense"
recorder sparse "$walk_sparse" --sparse 64,0.9
withheld_dense=0
withheld_sparse=0
case "$withheld" in
    dense) withheld_dense=1 ;;
    sparse) withheld_sparse=1 ;;
esac
if [ "$recorder_failed" != "0" ]; then
    if [ "$withheld_dense" = "1" ] && [ "$withheld_sparse" = "1" ]; then
        echo "the re-cut stops: both walks are short of launches the model makes" >&2
        exit 1
    fi
    echo "the re-cut withholds $withheld only: its walk is short of launches the model makes" >&2
fi

# 3. The blocked block, pasted from the tool rather than written by hand. A stale exemption is as wrong
#    as a missing one and harder to see, because a test comparing failures to the block goes green on a
#    file that exempts what nothing fails on.
echo 'declared blocked:'
: >"$staging/blocked-dense.txt"
: >"$staging/blocked-sparse.txt"
if [ "$withheld_dense" = "0" ]; then
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p program-server --example key_coverage -- \
        --fixture "$walk_dense" --target sm_100a --declared-blocked \
        2>/dev/null | grep '^# blocked:' >"$staging/blocked-dense.txt" || true
fi
if [ "$withheld_sparse" = "0" ]; then
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p program-server --example key_coverage -- \
        --fixture "$walk_sparse" --target sm_100a --declared-blocked \
        2>/dev/null | grep '^# blocked:' >"$staging/blocked-sparse.txt" || true
fi
printf '  dense %s, sparse %s\n' \
    "$(wc -l <"$staging/blocked-dense.txt")" "$(wc -l <"$staging/blocked-sparse.txt")"

# 4. The fixtures: the header, the block and the rows, from the walk's own print.
echo 'fixtures:'
: >"$fixture_dense"
: >"$fixture_sparse"
if [ "$withheld_dense" = "0" ]; then
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p minimax-h3 --example h3_fixture -- \
        --walk "$walk_dense" --commit "$commit" --blocked "$staging/blocked-dense.txt" >"$fixture_dense"
fi
if [ "$withheld_sparse" = "0" ]; then
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p minimax-h3 --example h3_fixture -- \
        --walk "$walk_sparse" --commit "$commit" --blocked "$staging/blocked-sparse.txt" >"$fixture_sparse"
fi

# 5. Certify: the fixture must be the walk's rows, exactly, and every key must be covered or declared
#    blocked. A fixture that fails either is one a store must not be built from.
echo 'certify:'
status=0
for pair in "dense:$fixture_dense:$walk_dense" "sparse:$fixture_sparse:$walk_sparse"; do
    name=${pair%%:*}
    rest=${pair#*:}
    fixture=${rest%%:*}
    walk=${rest#*:}

    # A withheld deployment has no fixture to certify, and its shortfall is already stated above.
    if [ "$(eval "echo \$withheld_$name")" = "1" ]; then
        printf '  %-10s withheld — the walk is short\n' "$name"
        status=1
        continue
    fi

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
    # `key_coverage` states a clean set two ways and only one of them counts: the clean run prints
    # `# every key is covered for <target>` and a short one prints `# N of M keys are not covered`.
    # Accepting only the second form's `0` made every clean fixture fail this gate - the branch
    # below then looked for a `missing-` list, found the empty file a clean run leaves, and reported
    # `uncovered keys and no block to excuse them` on a set with nothing uncovered.
    case "$coverage" in
        *"every key is covered"*|*" 0 of "*) ;;
        *)
            if [ ! -s "$staging/missing-$name.txt" ]; then
                echo "  $name has uncovered keys and no block to excuse them" >&2
                exit 1
            fi
            ;;
    esac
done

# 5. Move into place. Each fixture moved here is certified: `dense` and `sparse` are two deployments'
#    row sets at one commit, and one may be certified while the other is withheld, so each is moved on
#    its own evidence. What must not happen is a file moved on a certification that failed, and that
#    cannot: the loop above skips a withheld deployment and sets the status for it.
attempted=
if [ "$withheld_dense" = "0" ]; then
    attempted="$attempted dense"
fi
if [ "$withheld_sparse" = "0" ]; then
    attempted="$attempted sparse"
fi
if [ "$status" -ne 0 ]; then
    if [ -z "$attempted" ]; then
        echo 'a fixture failed certification; notes/programs/ was not touched'
        exit 1
    fi
    echo 'a fixture failed certification; the certified ones below still move' >&2
fi
for name in $attempted; do
    case "$name" in
        sparse)
            # One file holds the union: the sparse deployment launches every row the dense one does,
            # and a store built from a dense-only file lacked the rows a sparse run launches.
            cp "$fixture_sparse" notes/programs/minimax-h3-keys.txt
            echo 'written: notes/programs/minimax-h3-keys.txt (the union, stated as the sparse deployment)'
            ;;
        dense)
            # A dense certification states nothing the file lacks: its rows are the sparse set's
            # subset, so rewriting the file from it would drop the rows the sparse deployment launches.
            echo 'dense: certified; notes/programs/minimax-h3-keys.txt holds the union and is unchanged'
            ;;
    esac
done
if [ "$status" -ne 0 ]; then
    exit 1
fi
