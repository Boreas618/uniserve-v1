#!/bin/bash
# Runs TLC on one configuration and prints its verdict: the violated property
# with the actions of the shortest behaviour, or the states of a complete search.
#   tlc.sh MODULE [CONFIG] [-- extra TLC arguments]
# With RULES=NoRn the configuration's rules are replaced by that definition of
# the module: the mutation that leaves rule Rn out. With HAZARDS=Hn the hazard
# Hn of the system as built is switched on. With ONLY="Inv..." the configuration's
# invariants are replaced by TypeOK and the ones named.
# The full output is kept in $TLC_LOGS (default: a temporary directory); TLC's
# states are removed when the run ends.
set -u
T=${TLA_TOOLS:-/workspace/dev/tools/tla}
module=$1; config=${2:-$1}; shift; [ $# -gt 0 ] && shift; [ "${1:-}" = "--" ] && shift
logs=${TLC_LOGS:-$(mktemp -d)}; mkdir -p "$logs"
# The configurations are run from configs/; the modules they extend lie in
# model/, properties/ and proof/, which TLC searches as its library.
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here/configs"
spec="$module.tla"; [ -f "$spec" ] || spec="$here/model/$module.tla"
cfg="$here/configs/$config.cfg"; name="$config"
if [ -n "${RULES:-}" ]; then
    name="$name-$RULES"; sed "s/Rules <- AllRules/Rules <- $RULES/" "$cfg" > "$logs/$name.cfg"; cfg="$logs/$name.cfg"
fi
if [ -n "${HAZARDS:-}" ]; then
    name="$name-$HAZARDS"; sed "s/Hazards = {}/Hazards = {\"$HAZARDS\"}/" "$cfg" > "$logs/$name.cfg"; cfg="$logs/$name.cfg"
fi
if [ -n "${ONLY:-}" ]; then
    # The configuration's invariants are replaced by TypeOK and the named ones.
    name="$name-$(echo $ONLY | tr ' ' '-')"
    awk -v only="$ONLY" '/^INVARIANTS/ { print; print "    TypeOK " only; skip = 1; next }
                         skip && /^    / { next } { skip = 0; print }' "$cfg" > "$logs/$name.cfg"
    cfg="$logs/$name.cfg"
fi
log="$logs/$name.log"
"$T"/jdk-21.0.13+9/bin/java -XX:+UseParallelGC -Xmx${TLC_HEAP:-32g} \
    -DTLA-Library="$here/model:$here/properties:$here/proof:$T/tlaps/lib/tlapm/stdlib" -cp "$T/tla2tools.jar" tlc2.TLC \
    -workers ${TLC_WORKERS:-32} -noGenerateSpecTE -metadir "$logs/states-$name" -config "$cfg" "$@" "$spec" > "$log" 2>&1
status=$?
# The states directory is only TLC's working store.
rm -rf "$logs/states-$name"
grep -E "^Error: (Invariant|Action property|Temporal|Deadlock|.*is violated)|^State [0-9]+: <|states generated,|Model checking completed|The depth of the complete|Finished in|Error: " "$log" \
    | sed -E 's/^(State [0-9]+: <[A-Za-z]+(\([^)]*\))?).*/\1>/'
echo "exit $status, log $log"
exit $status
