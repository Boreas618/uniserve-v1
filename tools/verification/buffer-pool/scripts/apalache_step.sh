#!/bin/bash
# The inductive step of IndInv under Apalache, one process per group of actions
# (Next1 to Next11 of MC_Apalache). Prints one line per group: NoError, or the
# violated invariant's number; the counterexamples are under OUT/<group>.
#   apalache_step.sh OUT [GROUP...]
out=$(mkdir -p "$1" && cd "$1" && pwd); shift
groups=${*:-1 2 3 4 5 6 7 8 9 10 11}
: > "$out/result.txt"
for n in $groups; do
    ( JVM_ARGS="-Xmx24g" APALACHE_WORK="$out/modules-$n" "$(dirname "$0")"/apalache.sh --init=IndInit --next=Next$n \
          --inv=IndInv --length=1 --cinit=ConstInit --tuning-options='search.invariantFilter=1->.*' \
          --out-dir="$out/$n" > "$out/$n.log" 2>&1
      echo "Next$n: $(grep -E 'The outcome is|invariant [0-9]+ violated' "$out/$n.log" | tr -s ' ' | cut -c1-60 | tr '\n' ';')" >> "$out/result.txt" ) &
done
wait
sort -V "$out/result.txt"
