#!/bin/bash
# The suite of the buffer pool's design model: every configuration of the design
# completes with no violation, every rule left out violates a property, and every
# hazard of the system as built that is modelled violates one; z3 answers
# `unsat` to every goal of the address arithmetic; the pinned properties match
# their pins. Prints one line per run and exits non-zero when a run differs
# from what is stated here.
#   suite.sh [pins|design|mutations|hazards|validation|addresses]...      (all six by default)
set -u
cd "$(dirname "$0")"
export TLC_LOGS=${TLC_LOGS:-$(mktemp -d)}
export TLC_WORKERS=${TLC_WORKERS:-24}
parts=${*:-pins design mutations hazards validation addresses}
mkdir -p "$TLC_LOGS"
results="$TLC_LOGS/suite.txt"; : > "$results"

# The configurations of the design.
design="MC_Core MC_Copies MC_Handoff MC_Trial MC_TrialTiers MC_Resize MC_Tiers MC_Loss MC_LossCopies MC_Finality MC_Cross MC_HostParts MC_HostSplit"

# configuration, rule left out, the invariants one of which is reported.
mutations="
MC_Core NoR1 NoLeak
MC_Core NoR2 FreeUnheld
MC_Core NoR3 WriteIsolation|ReadIsolation
MC_Core NoR4 KeyStable
MC_Copies NoR5 Exclusive|KeyStable
MC_Core NoR6 WriteIsolation|SharedUnwritten
MC_Core NoR7 KeyStable
MC_Trial NoR8 TrialVisibility
MC_TrialTiers NoR8 TrialVisibility
MC_Handoff NoR9 TaskIsolation
MC_Tiers NoR9 TaskIsolation
MC_Resize NoR10 OneRegion
MC_Cross NoR5 TaskIsolation
MC_HostParts NoR11 CopiesPlaced
MC_HostSplit NoR11 CopiesPlaced
"

# configuration, hazard switched on, the properties one of which is reported.
hazards="
MC_Handoff H1 TaskIsolation
MC_Copies H3 CopyIsolation
MC_Tiers H4 Exactness
MC_Copies H5 CopyIsolation|FreeUnheld
MC_Finality H6 TaskFinality
MC_Trial H7 WriteIsolation|ReadIsolation
MC_Core H8 WriteIsolation|ReadIsolation
MC_Trial H9 Exclusive|WriteIsolation|ReadIsolation
MC_Core H10 WriteIsolation|SharedUnwritten
MC_Core H11 Exactness
MC_Core H12 NoLeak
MC_Finality H13 CopyFinality
MC_Cross H14 ReadIsolation
"

# configuration, rule left out or hazard switched on, and a property checked
# alone, which must be violated: the behaviour each property was admitted with
# where the runs above report another property first.
validation="
MC_Core H12 CapacityIdentity
MC_Core NoR2 CapacityIdentity
MC_Cross NoR3 CrossingsPlaced
MC_Cross NoR9 CrossingTaskIsolation
MC_HostSplit NoR5 CopiesPlaced
MC_Handoff NoR3 ImportsPlaced
"

# Runs one configuration and records whether it ended as `expected`: "none" for
# a complete search without a violation, else a pattern of invariant names.
run() {
    local label=$1 config=$2 expected=$3 output violated
    output=$(./tlc.sh "$config" 2>&1)
    violated=$(echo "$output" | sed -nE 's/^Error: (Invariant|Temporal property) (.*) (is|was) violated.*/\2/p' | head -1)
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
        pins)
            if ./pins.py check > "$TLC_LOGS/pins.txt" 2>&1; then echo "ok    pins: $(cat "$TLC_LOGS/pins.txt")" >> "$results"
            else echo "FAIL  pins: $(tr '\n' ';' < "$TLC_LOGS/pins.txt")" >> "$results"; fi ;;
        design)
            for config in $design; do run "$config" "$config" none & done ;;
        mutations)
            while read -r config rule expected; do
                [ -n "$config" ] && RULES=$rule run "$config $rule" "$config" "$expected"
            done <<< "$mutations" ;;
        hazards)
            while read -r config hazard expected; do
                [ -n "$config" ] && HAZARDS=$hazard run "$config $hazard" "$config" "$expected"
            done <<< "$hazards" ;;
        validation)
            while read -r config change property; do
                [ -n "$config" ] || continue
                if [ "${change#NoR}" != "$change" ]; then
                    RULES=$change ONLY=$property run "$config $change $property" "$config" "$property"
                else
                    HAZARDS=$change ONLY=$property run "$config $change $property" "$config" "$property"
                fi
            done <<< "$validation" ;;
        addresses)
            answers=$("${TLA_TOOLS:-/workspace/dev/tools/tla}"/tlaps/lib/tlapm/backends/bin/z3 ../proof/addresses.smt2 | sort | uniq -c | tr -s ' ' | tr '\n' ';')
            if [ "$answers" = " 13 unsat;" ]; then echo "ok    addresses: 13 goals unsat" >> "$results"
            else echo "FAIL  addresses: expected 13 unsat, got:$answers" >> "$results"; fi ;;
    esac
done
wait
sort "$results"
! grep -q "^FAIL" "$results"
