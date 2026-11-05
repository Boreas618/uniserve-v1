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
# The delivery's four durations, as the clips the model makes of them: `uniserve_eval/workloads/minimax_h3/`
# asks for 5, 8, 10 and 15 seconds, which at 24 fps are 120, 192, 240 and 360 frames and, through
# `align_num_frames`, clips of 124, 192, 243 and 362. **A duration is the delivery's; the clip is the
# model's.** The values are stated here rather than derived because this is `sh`; the generator states
# the same four through the two functions (`DELIVERY_SECONDS`), and a mismatch is a finding.
samples=124,192,243,362
# The six canvases the model serves, in the walk's own `HxW` order. **The authority is
# `crates/models/minimax-h3/src/config/canvas.rs`'s `NAMED_CANVASES`**, and this line is a second
# statement of it — `programs --rasters offered` reading `offered_canvases()` is owed so that a
# canvas added to the model reaches every fixture without anyone editing this file. Until then, a
# raster that is not in the model must not appear here: 480x832 was in this line for days and every
# fixture it produced named a canvas no request asks for.
rasters=672x1536,768x1344,768x1024,768x768,1024x768,1344x768
tokens=4096
conditioned=true
# The task the fixture's deployment serves. The walk refuses without it, and it is not cosmetic: the
# serving partition derives the capacity from it, so the packed row bound — and the gated activation's
# `wide` form with it — follows from this line. `ref2va` is the widest of the three: 304,976 dense and
# 351,232 sparse, and one store means the widest.
task=ref2va
# The tasks **one store serves**, which is every task the model answers: a row is the pair
# (operator, key) and the key depends on the task, so a walk of one task states one task's rows. The
# `logits_gather` that reads the modulation table is the one that differs — `columns_2` when nothing
# is conditioned over (`t2va`) and `columns_4` when a conditioner is (`fl2va`, `ref2va`) — so a
# fixture walked at `task` above lacks a cell a `t2va` request launches. The rows only the other
# tasks state are appended below, and their source is named in the fixture.
serves='t2va fl2va ref2va'
elements=123456
# One track per clip, `audio_samples` of it: the walk refuses a track no stated clip implies, so this
# line cannot drift from the one above without a refusal naming both.
track=165600,256000,324000,482400

# The sparse deployment's attention: 64 tiles of 0.9 budget, keyed at the deployment's own bound's next
# power of two. The bound is the deployment's, not one request's, and the walk refuses any other class
# by name.
sparse=64,0.9,8192

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

# 1. The walks. No `--rows` and no `--dtype`: the deployment's own bound and each component's own
#    type, both read from the checkpoint (or from `--types` where this host has no shards).
echo 'walk:'
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" \
    --samples "$samples" --rasters "$rasters" --tokens "$tokens" \
    --tp 1 --task "$task" --conditioned "$conditioned" --elements "$elements" --track "$track" \
    --types "$types" >"$walk_dense"
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" \
    --samples "$samples" --rasters "$rasters" --tokens "$tokens" \
    --tp 1 --task "$task" --conditioned "$conditioned" --elements "$elements" --track "$track" \
    --types "$types" --sparse "$sparse" >"$walk_sparse"
printf '  dense %s rows, sparse %s rows\n' \
    "$(grep -c '^[a-z]' "$walk_dense")" "$(grep -c '^[a-z]' "$walk_sparse")"

# 1b. The rows the other served tasks state and this one does not. **The walk states every served
# task's rows itself** - its gated activation and its logits gather iterate the deployment's task set -
# so this step appends nothing and refuses instead: a difference here means the walk narrowed to one
# task, and a fixture that silently patched it would be right by this tool rather than by the model,
# which is the form that survives longest and confuses the next person to re-check it.
echo 'rows the other served tasks state (must be zero - the walk states them):'
other_dense="$staging/walk-other-dense.txt"
other_sparse="$staging/walk-other-sparse.txt"
: >"$other_dense"
: >"$other_sparse"
for other in $serves; do
    [ "$other" = "$task" ] && continue
    case "$other" in
        t2va) other_conditioned=false ;;
        *) other_conditioned=true ;;
    esac
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p minimax-h3 --example programs -- "$root" \
        --samples "$samples" --rasters "$rasters" --tokens "$tokens" \
        --tp 1 --task "$other" --conditioned "$other_conditioned" --elements "$elements" \
        --track "$track" --types "$types" >>"$other_dense"
    RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
        cargo run -q -p minimax-h3 --example programs -- "$root" \
        --samples "$samples" --rasters "$rasters" --tokens "$tokens" \
        --tp 1 --task "$other" --conditioned "$other_conditioned" --elements "$elements" \
        --track "$track" --types "$types" --sparse "$sparse" >>"$other_sparse"
done
# Only whole rows (`operator<TAB>key`), only those the fixture's own task does not state, sorted and
# deduplicated across the other tasks.
for pair in "dense:$walk_dense:$other_dense" "sparse:$walk_sparse:$other_sparse"; do
    name=${pair%%:*}
    rest=${pair#*:}
    walk=${rest%%:*}
    others=${rest#*:}
    only="$staging/only-$name.txt"
    comm -23 \
        <(grep -P '^[a-z0-9_]+\t' "$others" | sort -u) \
        <(grep -P '^[a-z0-9_]+\t' "$walk" | sort -u) >"$only"
    printf '  %-10s %s row(s)' "$name" "$(wc -l <"$only")"
    if [ -s "$only" ]; then
        printf ' - FAILED, the walk does not state them:\n'
        sed 's/^/    /' "$only"
        exit 1
    fi
    printf '\n'
done

# 2. The blocked block, pasted from the tool rather than written by hand. A stale exemption is as wrong
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

# 3. The fixtures: the header, the block and the rows, from the walk's own print.
echo 'fixtures:'
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example h3_fixture -- \
    --walk "$walk_dense" --commit "$commit" --blocked "$staging/blocked-dense.txt" >"$fixture_dense"
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example h3_fixture -- \
    --walk "$walk_sparse" --commit "$commit" --blocked "$staging/blocked-sparse.txt" >"$fixture_sparse"

# 3b. One file, one attention backend. The dense and sparse walks differ by seven operators
# (`sparse_attention` and the six `vsa_*`), so a dense fixture that names one of them is the union of
# two deployments - a store with cells no launch of that deployment reads, and the omni fixture was
# deleted for exactly this. The check is a grep because the alternative is a rule nobody runs.
echo 'one deployment per file:'
for pair in "dense:$fixture_dense:" "sparse:$fixture_sparse:sparse"; do
    name=${pair%%:*}
    rest=${pair#*:}
    fixture=${rest%%:*}
    want=${rest#*:}
    others=$(grep -cP '^(sparse_attention|vsa_[a-z_]+)\t' "$fixture")
    if [ "$name" = dense ] && [ "$others" -ne 0 ]; then
        printf '  %-10s FAILED - %s row(s) of the sparse backend in the dense file\n' "$name" "$others"
        exit 1
    fi
    if [ "$name" = sparse ] && [ "$others" -eq 0 ]; then
        printf '  %-10s FAILED - no row of the sparse backend in the sparse file\n' "$name"
        exit 1
    fi
    printf '  %-10s ok (%s row(s) of the sparse backend)\n' "$name" "$others"
done

# 4. Certify: the fixture must be the walk's rows, exactly, and every key must be covered or declared
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
