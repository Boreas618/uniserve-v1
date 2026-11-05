#!/bin/sh
# Record what the deployment lists were checked against, at one tip, as evidence.
#
# `tools/check_h3_deployment.sh` answers eight questions now, about the tree it ran in.
# This writes that answer down where a later reader can find it: the tip, the two
# fixtures' own identities, the command, every line the check printed, and its exit
# code. A post and a re-run both answer today's question; the artifact answers the
# question that was asked, and names the file it was asked about.
#
# It is not a gate. It exits with the check's own code, so a caller may gate on it,
# but nothing here decides anything on its own: the run's log is the record and the
# check's eight questions are the evidence.
#
# Usage:
#   tools/certify_h3.sh <checkpoint root> <element-types json> [<directory>]
#
#   <directory>   where the record goes; default `artifacts/h3-certification/<tip>`
#
# What it writes there:
#   check.log      every line the check printed, unedited
#   record.txt     the identity of what was checked, and the answers, one per line
set -eu

root=${1:?the checkpoint root is the first argument}
types=${2:?the element-types json is the second argument}
here=$(cd "$(dirname "$0")/.." && pwd)

tip=$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo unstated)
out=${3:-$here/artifacts/h3-certification/$tip}
mkdir -p "$out"

dense=$here/notes/programs/minimax-h3-keys.txt
sparse=$here/notes/programs/minimax-h3-sparse-keys.txt

# A count is of the copy it read: two checkouts of one path hold two files, and the
# fixture's own `# commit:` line states the tree its rows describe, which is not
# always the tip the check runs at. Both are recorded, beside a digest of the bytes.
digest() {
    if [ -f "$1" ]; then
        sha256sum "$1" | cut -d' ' -f1
    else
        echo absent
    fi
}

rows() {
    if [ -f "$1" ]; then
        grep -cP '\t' "$1" || true
    else
        echo 0
    fi
}

stamp() {
    if [ -f "$1" ]; then
        grep -m1 '^# commit:' "$1" | sed 's/^# commit: *//' || echo unstated
    else
        echo absent
    fi
}

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
sh "$here/tools/check_h3_deployment.sh" "$root" "$types" >"$out/check.log" 2>&1
status=$?
set -e
ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
    echo "# H3 deployment certification"
    echo
    echo "This is a record of one run of \`tools/check_h3_deployment.sh\`, not an"
    echo "acceptance gate. The run's own lines are in \`check.log\` beside this file;"
    echo "the exit code below is the check's, unmodified."
    echo
    echo "tip: $tip"
    echo "started: $started"
    echo "ended: $ended"
    echo "exit: $status"
    echo "command: sh tools/check_h3_deployment.sh $root $types"
    echo "checkpoint root: $root"
    # The element-types table is an input the eight questions read and this lane does not own:
    # every `dtype` answer moves together if it drifts, so a record that pinned only the tip would
    # call two runs with different tables identical. Pinned by content, like the rows.
    echo "element types: $types"
    echo "element-types bytes: $(wc -c <"$types" | tr -d ' ') sha256: $(digest "$types")"
    echo
    echo "## the two files, as this run read them"
    echo
    printf '%-8s %-6s %-12s %-64s %s\n' file rows "row stamp" sha256 path
    printf '%-8s %-6s %-12s %-64s %s\n' dense "$(rows "$dense")" "$(stamp "$dense")" "$(digest "$dense")" "$dense"
    printf '%-8s %-6s %-12s %-64s %s\n' sparse "$(rows "$sparse")" "$(stamp "$sparse")" "$(digest "$sparse")" "$sparse"
    echo
    echo "## the eight answers, as the run printed them"
    echo
    grep -E '^(the list|   [0-9]+ (row|of)|   [a-z]+: |   24 point)' "$out/check.log" || true
} >"$out/record.txt"

cat "$out/record.txt"
exit "$status"
