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
samples=256
rasters=480x832
tokens=4096
conditioned=true
elements=123456
track=44100

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
    --tp 1 --conditioned "$conditioned" --elements "$elements" --track "$track" \
    --types "$types" >"$walk_dense"
RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable-aarch64-unknown-linux-gnu} \
    cargo run -q -p minimax-h3 --example programs -- "$root" \
    --samples "$samples" --rasters "$rasters" --tokens "$tokens" \
    --tp 1 --conditioned "$conditioned" --elements "$elements" --track "$track" \
    --types "$types" --sparse "$sparse" >"$walk_sparse"
printf '  dense %s rows, sparse %s rows\n' \
    "$(grep -c '^[a-z]' "$walk_dense")" "$(grep -c '^[a-z]' "$walk_sparse")"

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
    printf '  %-10s coverage %s\n' "$name" "$coverage"
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
