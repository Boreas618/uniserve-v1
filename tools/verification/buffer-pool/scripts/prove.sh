#!/bin/bash
# Proves modules of proof/ with TLAPS and prints each one's verdict line.
#   prove.sh [MODULE...]
# The default is every module, in the order the proof builds on: KvProof, the
# shared lemmas KvFrame, the steps KvStep1 to KvStep11, their assembly KvStep,
# then KvProofP7 and KvSafeP7. Each runs in proof/, which keeps the fingerprints
# of proved obligations (.tlacache), with the modules of model/ and properties/
# on its include path, and its log is proof/<module>.log. TLAPS_THREADS sets the
# threads of each (default 48).
set -u
T=${TLA_TOOLS:-/workspace/dev/tools/tla}
export PATH=$T/tlaps/bin:$T/jdk-21.0.13+9/bin:$PATH
here=$(cd "$(dirname "$0")/.." && pwd)
modules=${*:-KvProof KvFrame KvStep1 KvStep2 KvStep3 KvStep4 KvStep5 KvStep6 KvStep7 KvStep8 KvStep9 KvStep10 KvStep11 KvStep KvProofP7 KvSafeP7}
cd "$here/proof"
status=0
for module in $modules; do
    tlapm -I ../model -I ../properties --threads "${TLAPS_THREADS:-48}" --stretch 6 "$module.tla" > "$module.log" 2>&1 || status=1
    echo "$module: $(grep -E 'obligations (proved|failed)' "$module.log" | tail -1)"
done
exit $status
