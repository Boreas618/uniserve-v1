#!/bin/sh
# The numeric block diagnosis for MiniMax-H3's stages 1 and 3, in one command.
#
# Design 10: a block-by-block comparison with today's implementation is a **diagnosis, never an
# acceptance gate**. This script produces the artifacts and names the first block whose error its
# own layer does not explain. Nothing here fails a build; a non-zero exit means an artifact could
# not be produced, never that a number is bad.
#
# Stage 1 is the text encoder (prompt -> hidden), stage 3 is the sampler (the sigma ladder the
# denoiser steps along). Neither needs the decoder, which is why they are the two stages a run can
# reach today.
#
# Usage:
#   tools/diagnose_h3_numeric.sh <checkpoint root> [<out dir>]
#
#   <checkpoint root>  e.g. /opt/uv/models/MiniMax-H3
#   <out dir>          default artifacts/h3-numeric
#
# Every artifact carries the commit and the checkpoint it was produced from: a sigma ladder with no
# commit beside it is a number nobody can date.

set -eu

root=${1:?usage: diagnose_h3_numeric.sh <checkpoint root> [<out dir>]}
out=${2:-artifacts/h3-numeric}

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

commit=$(git rev-parse --short HEAD 2>/dev/null || echo "uncommitted")
run="$out/$commit"
mkdir -p "$run"

echo "commit $commit, checkpoint $root, out $run"

# 1. Stage 3: the sampler's schedule. Host arithmetic on the checkpoint's own statement, so this
#    half runs with no device and no weights.
echo
echo "== stage 3: the sigma ladder and the network times"
cargo run -q -p minimax-h3 --example schedule_dump -- \
    --root "$root" --out "$run/schedule" --commit "$commit" | sed 's/^/   /'

echo
echo "== stage 3: ours against the reference's own evaluation of the same statement"
.venv/bin/python crates/models/minimax-h3/tools/compare_schedule.py \
    --dump "$run/schedule" > "$run/schedule-compare.txt" 2>&1 || true
tail -3 "$run/schedule-compare.txt" | sed 's/^/   /'

# 2. Stage 1: the text encoder. The load's own name check runs anywhere the index is; the numeric
#    half needs the shards and a device, so its two commands are printed rather than run.
echo
echo "== stage 1: the load's name check"
cargo run -q -p minimax-h3 --example encoder -- \
    --stage weights --root "$root" 2>&1 | grep -E '^(names|io):' | sed 's/^/   /' || true

cat >"$run/stage1-commands.txt" <<COMMANDS
Stage 1 (the text encoder) numeric half: two commands, both on the host that has the shards and a
device. They produce two dump trees that tools/diagnosis/compare.py compares by name.

  # ours
  cargo run -p minimax-h3 --example encoder -- --stage encode --root $root \\
      --device 0 --store <store> --out $run/rust

  # the reference's, for the same prompt and ids
  .venv/bin/python crates/models/minimax-h3/tools/dump_encoder.py --root $root \\
      --case <case.json> --probe prompt_hidden --out $run/ref

  # the comparison: per tensor relative L2, max abs, cosine, and whether the two are identical
  .venv/bin/python tools/diagnosis/compare.py --ref $run/ref --got $run/rust \\
      --out $run/report.json

Recorded here rather than run because this host has the text encoder's index and none of its shards:
the load's name check passes (see above) and the first payload read stops.
COMMANDS
sed 's/^/   /' "$run/stage1-commands.txt" | head -3

# 3. The reading. A block is explained by its layer when the difference is at that layer's rounding:
#    float32 evaluation order on a value near 1 is ~1.2e-07 absolute, and a table of indices is
#    explained only when it is identical.
echo
echo "== the first block whose error its layer does not explain"
{
    if grep -q '^verdict .*no .*differ' "$run/schedule-compare.txt" 2>/dev/null; then
        echo "   none in stage 3: the schedule agrees."
    elif grep -q 'worst ' "$run/schedule-compare.txt" 2>/dev/null; then
        worst=$(grep -m1 'worst ' "$run/schedule-compare.txt")
        largest=$(sed -n 's/.*abs \([0-9.e+-]*\).*/\1/p' "$run/schedule-compare.txt" | sort -g | tail -1)
        echo "   stage 3, $worst"
        echo "   largest absolute difference $largest"
        case $largest in
        1.[0-9]*e-0[6-7] | [0-9].[0-9]*e-0[6-7] | 0.000000*)
            echo "   explained: a sigma evaluated in float32 differs in its last place, and the"
            echo "   schedule's structure (points, steps, grid kind) agrees. Not a defect."
            ;;
        *)
            echo "   NOT explained by float32 rounding: read the named index against the two"
            echo "   implementations' evaluation order before concluding anything."
            ;;
        esac
    else
        echo "   stage 3 produced no comparison; the artifact is $run/schedule-compare.txt"
    fi
    echo "   stage 1 has no numbers on this host: see $run/stage1-commands.txt"
} | tee "$run/reading.txt"

echo
echo "artifacts under $run: the schedule dump, its comparison, the stage-1 commands and the reading."
echo "This is a diagnosis (design 10): nothing above gates the branch."
