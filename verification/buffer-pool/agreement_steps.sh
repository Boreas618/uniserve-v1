#!/bin/bash
# L1 of specs/buffer-pool.md §19, step by step: every transition the
# enumeration of the Rust buffer pool reaches (a state, the input taken in
# it, the state it leads to) in the named configurations, `core` by default,
# is a path of MC_Pool's steps between the two states (steps.py,
# KvTrace.tla); and a transition made by hand that the model does not allow
# (the enumeration's forged.steps: a commit that takes effect before its
# stamp) is reported. Prints one line per check and exits non-zero when one
# fails. agreement.sh compares the sets of states; this compares the steps.
#   agreement_steps.sh [core|copies|transfer|tiers]...
# The enumeration and TLC's runs are kept in $AGREEMENT_WORK (default: a
# temporary directory).
set -u
cd "$(dirname "$0")"
work=${AGREEMENT_WORK:-$(mktemp -d)}
mkdir -p "$work/rust"

if ! (cd ../.. && UNISERVE_POOL_STATES="$work/rust" cargo test --release -p storage-server --lib \
        buffer_pool::enumeration -- --ignored --nocapture) > "$work/enumeration.log" 2>&1; then
    echo "FAIL  the enumeration of the Rust buffer pool: $(grep -m1 -E 'panicked|^error' "$work/enumeration.log")"
    exit 1
fi

status=0
for shape in ${*:-core}; do
    python3 steps.py "$work/rust/$shape.steps" "$work/$shape" || status=1
done
python3 steps.py --expect fail "$work/rust/forged.steps" "$work/forged" || status=1
exit $status
