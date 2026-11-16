#!/bin/bash
# The trace of a run of the Storage Server (specs/buffer-pool.md §18): each
# deployment's trace in TRACE_DIR (written by a Storage Server whose settings
# state `buffer_pool_trace = TRACE_DIR`) is replayed on a new Rust buffer pool
# (`buffer-pool-verify replay`), with
# every completion checked against the run's and the pool's check() run after
# every input (a difference or an inconsistency fails), and every transition
# between the states the pool passes through is checked against the design
# model's steps from its recorded state, on the parts of the pool it touches
# (steps.py, KvTrace.tla), $REPLAY_JOBS (default 32) scans and TLC runs at
# once. Prints one verdict line (ok, FAIL, or INCONCLUSIVE when a TLC run
# stopped before its search was complete) and exits non-zero unless ok.
#   replay.sh TRACE_DIR
# The replay's states and TLC's runs are kept in $REPLAY_WORK (default: a
# temporary directory; the states of a serving run take gigabytes, so a local
# disk), TLC's state files under $TMPDIR (default /tmp).
set -u
here=$(cd "$(dirname "$0")" && pwd)
trace=$(cd "$1" && pwd)
work=${REPLAY_WORK:-$(mktemp -d)}
mkdir -p "$work/states"

if ! (cd "$here/../../.." && cargo run --release -p storage-server --features verification \
        --bin buffer-pool-verify -- replay "$trace" --states "$work/states") \
        > "$work/replay.log" 2>&1; then
    echo "FAIL  replay of $trace: $(tail -1 "$work/replay.log" | cut -c1-600)"
    exit 1
fi
replayed=$(grep -E "^replayed " "$work/replay.log" | sed 's/^replayed //' | paste -sd ';' -)

status=0
failed=0
model=""
for steps in "$work"/states/*.steps; do
    name=$(basename "$steps" .steps)
    result=$(python3 "$here/steps.py" --jobs "${REPLAY_JOBS:-32}" "$steps" "$work/$name") || status=1
    # A TLC run that stopped before its search was complete settles nothing.
    case "$result" in INCONCLUSIVE*) ;; ok*) ;; *) failed=1 ;; esac
    model="$model $(echo "$result" | sed -E 's/^(ok|FAIL|INCONCLUSIVE) +//')"
done
if [ $status = 0 ]; then verdict=ok; elif [ $failed = 0 ]; then verdict=INCONCLUSIVE; else verdict=FAIL; fi
echo "$verdict    replay: $replayed; model:$model"
exit $status
