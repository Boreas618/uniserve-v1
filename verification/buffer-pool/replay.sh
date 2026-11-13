#!/bin/bash
# The trace of a run of the Storage Server (specs/buffer-pool.md §18): each
# deployment's trace in TRACE_DIR (written with `storage-server serve
# --buffer-pool-trace TRACE_DIR`) is replayed on a new Rust buffer pool, with
# every completion checked against the run's, and the states the pool passes
# through are checked against the design model's steps in windows of at most
# WINDOW steps, each from a recorded state (steps.py, KvTrace.tla). Prints one
# verdict line and exits non-zero when either fails.
#   replay.sh TRACE_DIR [WINDOW]
# The replay's states and TLC's runs are kept in $REPLAY_WORK (default: a
# temporary directory).
set -u
here=$(cd "$(dirname "$0")" && pwd)
trace=$(cd "$1" && pwd)
work=${REPLAY_WORK:-$(mktemp -d)}
mkdir -p "$work/states"

if ! (cd "$here/../.." && UNISERVE_POOL_TRACE="$trace" UNISERVE_POOL_STATES="$work/states" \
        cargo test --release -p storage-server --lib buffer_pool::trace -- --ignored --nocapture) \
        > "$work/replay.log" 2>&1; then
    echo "FAIL  replay of $trace: $(grep -m1 -A1 -E 'panicked|^error' "$work/replay.log" | tr '\n' ' ' | cut -c1-600)"
    exit 1
fi
replayed=$(grep -E "^replayed " "$work/replay.log" | sed 's/^replayed //' | paste -sd ';' -)

status=0
model=""
for steps in "$work"/states/*.steps; do
    name=$(basename "$steps" .steps)
    result=$(python3 "$here/steps.py" --chain "${2:-50}" "$steps" "$work/$name") || status=1
    model="$model $(echo "$result" | sed -E 's/^(ok|FAIL) +//')"
done
if [ $status = 0 ]; then verdict=ok; else verdict=FAIL; fi
echo "$verdict    replay: $replayed; model:$model"
exit $status
