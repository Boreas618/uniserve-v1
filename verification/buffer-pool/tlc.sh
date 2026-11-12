#!/bin/bash
# Runs TLC on one configuration and prints its verdict: the violated property
# with the actions of the shortest behaviour, or the states of a complete search.
#   tlc.sh MODULE [CONFIG] [-- extra TLC arguments]
# With RULES=NoRn the configuration's rules are replaced by that definition of
# the module: the mutation that leaves rule Rn out. With HAZARDS=Hn the hazard
# Hn of the system as built is switched on.
# The full output is kept in $TLC_LOGS (default: a temporary directory).
set -u
T=${TLA_TOOLS:-/workspace/dev/tools/tla}
module=$1; config=${2:-$1}; shift; [ $# -gt 0 ] && shift; [ "${1:-}" = "--" ] && shift
logs=${TLC_LOGS:-$(mktemp -d)}; mkdir -p "$logs"
cd "$(dirname "$0")"
cfg="$config.cfg"; name="$config"
if [ -n "${RULES:-}" ]; then
    name="$config-$RULES"; cfg="$logs/$name.cfg"
    sed "s/Rules <- AllRules/Rules <- $RULES/" "$config.cfg" > "$cfg"
elif [ -n "${HAZARDS:-}" ]; then
    name="$config-$HAZARDS"; cfg="$logs/$name.cfg"
    sed "s/Hazards = {}/Hazards = {\"$HAZARDS\"}/" "$config.cfg" > "$cfg"
fi
log="$logs/$name.log"
"$T"/jdk-21.0.13+9/bin/java -XX:+UseParallelGC -Xmx${TLC_HEAP:-32g} \
    -DTLA-Library="$T/tlaps/lib/tlapm/stdlib" -cp "$T/tla2tools.jar" tlc2.TLC \
    -workers ${TLC_WORKERS:-32} -noGenerateSpecTE -metadir "$logs/states-$name" -config "$cfg" "$@" "$module" > "$log" 2>&1
status=$?
grep -E "^Error: (Invariant|Action property|Temporal|Deadlock|.*is violated)|^State [0-9]+: <|states generated,|Model checking completed|The depth of the complete|Finished in|Error: " "$log" \
    | sed -E 's/^(State [0-9]+: <[A-Za-z]+(\([^)]*\))?).*/\1>/'
echo "exit $status, log $log"
exit $status
