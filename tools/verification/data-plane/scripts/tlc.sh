#!/bin/bash
# Runs TLC on one configuration of the data plane's model and prints its
# verdict: the violated property with the actions of the shortest behaviour, or
# the states of a complete search.
#   tlc.sh CONFIG [-- extra TLC arguments]
# With HAZARDS=Name the hazard of that name is switched on. With ONLY="Inv..."
# the configuration's invariants are replaced by TypeOK and the ones named.
# The full output and TLC's states are kept in $TLC_LOGS (default:
# artifacts/verification/data-plane under the repository, on the shared disk).
set -u
T=${TLA_TOOLS:-/workspace/dev/tools/tla}
config=$1; shift; [ "${1:-}" = "--" ] && shift
here=$(cd "$(dirname "$0")/.." && pwd)
logs=${TLC_LOGS:-$(cd "$here/../../.." && pwd)/artifacts/verification/data-plane}; mkdir -p "$logs"
# The configurations run from configs/; the modules they extend lie in model/
# and properties/, which TLC searches as its library.
cd "$here/configs"
cfg="$here/configs/$config.cfg"; name="$config"
if [ -n "${HAZARDS:-}" ]; then
    name="$name-$HAZARDS"; sed "s/Hazards = {}/Hazards = {\"$HAZARDS\"}/" "$cfg" > "$logs/$name.cfg"; cfg="$logs/$name.cfg"
fi
if [ -n "${ONLY:-}" ]; then
    name="$name-$(echo $ONLY | tr ' ' '-')"
    awk -v only="$ONLY" '/^INVARIANTS/ { print; print "    TypeOK " only; skip = 1; next }
                         skip && /^    / { next } { skip = 0; print }' "$cfg" > "$logs/$name.cfg.tmp"
    mv "$logs/$name.cfg.tmp" "$logs/$name.cfg"; cfg="$logs/$name.cfg"
fi
log="$logs/$name.log"
"$T"/jdk-21.0.13+9/bin/java -XX:+UseParallelGC -Xmx${TLC_HEAP:-32g} \
    -DTLA-Library="$here/model:$here/properties" -cp "$T/tla2tools.jar" tlc2.TLC \
    -workers ${TLC_WORKERS:-32} -noGenerateSpecTE -metadir "$logs/states-$name" -config "$cfg" "$@" "$config.tla" > "$log" 2>&1
status=$?
# The states directory is only TLC's working store: removed once the run ends.
rm -rf "$logs/states-$name"
grep -E "^Error: (Invariant|Action property|Temporal|Deadlock|.*is violated)|^State [0-9]+: <|states generated,|Model checking completed|The depth of the complete|Finished in|^Error: " "$log" \
    | sed -E 's/^(State [0-9]+: <[A-Za-z]+(\([^)]*\))?).*/\1>/'
echo "exit $status, log $log"
exit $status
