#!/bin/bash
# L1 of specs/buffer-pool.md §19: every request sequence of seven small
# configurations is enumerated on the Rust buffer pool, no sequence breaks an
# invariant, and every state reached is a state of the design model's buffer
# pool (MC_Pool under the configuration of the same name). Prints one line per
# configuration and exits non-zero when one fails.
#   agreement.sh [core|copies|transfer|tiers|cross|parts|trial]...      (all seven by default)
# The enumeration and the model's states are kept in $AGREEMENT_WORK (default:
# a temporary directory); a configuration's model states pass through a named
# pipe and are not stored.
set -u
cd "$(dirname "$0")"
work=${AGREEMENT_WORK:-$(mktemp -d)}
mkdir -p "$work/rust"

if ! (cd ../../.. && cargo run --release -p storage-server --features verification \
        --bin buffer-pool-verify -- enumerate --states "$work/rust") > "$work/enumeration.log" 2>&1; then
    echo "FAIL  the enumeration of the Rust buffer pool: $(tail -1 "$work/enumeration.log")"
    exit 1
fi
grep -E "^[a-z]+: [0-9]+ states" "$work/enumeration.log" | sed 's/^/      Rust buffer pool, /'

status=0
for shape in ${*:-core copies transfer tiers cross parts trial}; do
    config=MC_Pool$(echo "${shape:0:1}" | tr a-z A-Z)${shape:1}
    # A configuration with constants of its own has a module of its own over MC_Pool.
    module=MC_Pool; [ -f "$config.tla" ] && module=$config
    rm -f "$work/$shape.dump"; mkfifo "$work/$shape.dump"
    TLC_LOGS="$work/logs" ./tlc.sh "$module" "$config" -- -dump "$work/$shape" > "$work/$shape.tlc" 2>&1 &
    if result=$(python3 agreement.py "$shape" "$work/$shape.dump" "$work/rust/$shape.txt"); then
        verdict=ok
    else
        verdict=FAIL; status=1
    fi
    wait
    if ! grep -q "states generated" "$work/$shape.tlc"; then
        verdict=FAIL; status=1; result="TLC did not complete: $(tail -1 "$work/$shape.tlc")"
    fi
    echo "$verdict    $(echo "$result" | tail -n +1 | tr '\n' ' ' | cut -c1-400)"
    [ "$verdict" = ok ] || echo "$result" > "$work/$shape.failure"
    rm -f "$work/$shape.dump"
done
exit $status
