#!/bin/bash
# The suite of the data plane's model: every configuration of the design
# completes with no violation, every hazard violates the property stated for
# it, and every witness of reachability is violated. Prints one line per run
# and exits non-zero when a run differs from what is stated here.
#   suite.sh [design|hazards|validation|witnesses|finality]...      (all by default)
set -u
cd "$(dirname "$0")"
export TLC_LOGS=${TLC_LOGS:-$(cd ../../../.. && pwd)/artifacts/verification/data-plane/suite}
parts=${*:-design hazards validation witnesses finality}
mkdir -p "$TLC_LOGS"
results="$TLC_LOGS/suite.txt"; : > "$results"

# The configurations of the design, each searched completely, with the TLC
# workers each takes: MC_Switch, the largest, has most of the host.
design="MC_Handoff:8 MC_Stale:16 MC_Reuse:16 MC_Switch:96"

# configuration, hazard switched on, the invariants one of which is reported.
hazards="
MC_Handoff EarlyRelease LeaseHeld
MC_Handoff UnreservedPush Reserved
MC_Handoff KeyedTail UnkeyedPushed
MC_Reuse Unfenced Filled
MC_Stale StaleAddress Integrity|Filled
MC_Switch SeedBlind Filled
MC_Switch SeedlessKey Faithful|Filled
MC_Switch RoleFlush Holding|Reserved|LeaseHeld
"

# configuration, hazard, and properties checked alone, one of which must be
# violated: the wrong content itself where the run above reports a guard first.
validation="
MC_Handoff KeyedTail Filled|Faithful
MC_Reuse EarlyRelease Filled|Faithful
MC_Reuse UnreservedPush Filled|Faithful
MC_Switch RoleFlush Filled|Faithful
"

# configuration and the witnesses it must violate.
witnesses="
MC_Handoff NoPush NoRead NoPullHost NoPullStore NoUnkeyedRead NoStaleMiss NoExpiredRead
MC_Reuse NoReuse
MC_Switch NoSeedMix NoRoleDrain
"

# Runs one configuration and records whether it ended as `expected`: "none" for
# a complete search without a violation, else a pattern of property names.
run() {
    local label=$1 config=$2 expected=$3 output violated
    output=$(./tlc.sh "$config" 2>&1)
    violated=$(echo "$output" | sed -nE 's/^Error: (Invariant|Temporal property|Action property) (.*) (is|was) violated.*/\2/p' | head -1)
    if [ "$expected" = none ]; then
        if echo "$output" | grep -q "Model checking completed. No error has been found."; then
            echo "ok    $label: $(echo "$output" | grep 'distinct states found' | tail -1)" >> "$results"
        else
            echo "FAIL  $label: expected a complete search, got: ${violated:-$(echo "$output" | tail -1)}" >> "$results"
        fi
    elif [ -n "$violated" ] && echo "$violated" | grep -Eqx "$expected"; then
        echo "ok    $label: $violated" >> "$results"
    else
        echo "FAIL  $label: expected $expected, got: ${violated:-no violation}" >> "$results"
    fi
}

for part in $parts; do
    case $part in
        design)
            for entry in $design; do
                config=${entry%:*}; workers=${entry#*:}
                TLC_WORKERS=$workers TLC_HEAP=$((workers * 2))g run "$config" "$config" none &
            done ;;
        hazards)
            while read -r config hazard expected; do
                [ -n "$config" ] && TLC_WORKERS=4 TLC_HEAP=8g HAZARDS=$hazard run "$config $hazard" "$config" "$expected" &
            done <<< "$hazards" ;;
        validation)
            while read -r config hazard expected; do
                [ -n "$config" ] && TLC_WORKERS=4 TLC_HEAP=8g HAZARDS=$hazard ONLY="${expected//|/ }" \
                    run "$config $hazard ${expected//|/ }" "$config" "$expected" &
            done <<< "$validation" ;;
        witnesses)
            while read -r config names; do
                [ -n "$config" ] || continue
                for name in $names; do
                    TLC_WORKERS=4 TLC_HEAP=8g ONLY=$name run "$config $name" "$config" "$name" &
                done
            done <<< "$witnesses" ;;
        finality)
            TLC_WORKERS=8 run MC_Finality MC_Finality none & ;;
    esac
done
wait
sort "$results"
! grep -q "^FAIL" "$results"
