#!/bin/sh
# The numeric block diagnosis for MiniMax-H3's stages 1 and 3, in one command.
#
# Design 10: a block-by-block comparison with today's implementation is a **diagnosis, never an
# acceptance gate**. This script produces the artifacts and names the first block whose error its
# own layer does not explain. Nothing here fails a build; a non-zero exit means an artifact could
# not be produced, never that a number is bad.
#
# The stages are the canon's (`notes/h3/diagnosis.md`), and a number is quoted with its name: stage 1
# is the text encoder (prompt -> hidden), stage 2 the video decoder, stage 3 the sampler (the sigma
# ladder the denoiser steps along), stage 4 the video encoder and stage 5 the conditioned path
# (`fl2va`) that binds it. **The denoiser's own blocks have no canon number**: stage 3 exercises them
# at its ladder's sigmas, so this script names them by name and never "stage 2". Stages 4 and 5
# compare **the encoder's own rows**, not the projected prefix: a difference past the projection
# cannot say whether the encoder or the projection made it.
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

# The reference half reads the retired Python tree; `UNISERVE_REFERENCE_TREE` names a checkout of it.
reference=${UNISERVE_REFERENCE_TREE:-/workspace/uniserve-v1}
reference_python="env PYTHONPATH=$reference:$reference/uniserve_kernels/src $reference/.venv/bin/python"

# 1. Stage 3: the sampler's schedule. Host arithmetic on the checkpoint's own statement, so this
#    half runs with no device and no weights.
echo
echo "== stage 3: the sigma ladder and the network times"
cargo run -q -p minimax-h3 --example schedule_dump -- \
    --root "$root" --out "$run/schedule" --commit "$commit" | sed 's/^/   /'

echo
echo "== stage 3: ours against the reference's own evaluation of the same statement"
$reference_python crates/models/minimax-h3/tools/compare_schedule.py \
    --dump "$run/schedule" > "$run/schedule-compare.txt" 2>&1 || true
tail -3 "$run/schedule-compare.txt" | sed 's/^/   /'

# 2. Stage 1: the text encoder. The load's own name check runs anywhere the index is; the numeric
#    half needs the shards and a device, so its two commands are printed rather than run.
echo
echo "== before stage 1: the load's name check (a precondition of stage 1, not a stage)"
cargo run -q -p minimax-h3 --example encoder -- \
    --stage weights --root "$root" 2>&1 | grep -E '^(names|io):' | sed 's/^/   /' || true

cat >"$run/stage1-commands.txt" <<COMMANDS
Stage 1 (the text encoder) numeric half: two commands, both on the host that has the shards and a
device. They produce two dump trees that tools/diagnosis/compare.py compares by name.

  # ours
  cargo run -p minimax-h3 --example encoder -- --stage encode --root $root \\
      --device 0 --store <store> --out $run/rust

  # the reference's, for the same prompt and ids
  $reference_python crates/models/minimax-h3/tools/dump_encoder.py --root $root \\
      --case <case.json> --probe prompt_hidden --out $run/ref

  # the comparison: per tensor relative L2, max abs, cosine, and whether the two are identical
  .venv/bin/python tools/diagnosis/compare.py --ref $run/ref --got $run/rust \\
      --out $run/report.json

Recorded here rather than run because this host has the text encoder's index and none of its shards:
the load's name check passes (see above) and the first payload read stops.
COMMANDS
sed 's/^/   /' "$run/stage1-commands.txt" | head -3
# 2b. The denoiser's own blocks - no canon number, exercised by stage 3 (the sampler) at its
#     ladder's sigmas. The probe catalogue is read from the harness that draws the reference, so the
#     list and the probes cannot drift; the host-only probes are attempted when a case is given, and
#     whatever stops them is named rather than assumed away. The device half needs the run host and
#     is printed as the name check's is.
echo
echo "== the denoiser's blocks (stage 3's subject, no canon number of its own): the probes"
$reference_python crates/models/minimax-h3/tools/dump_denoiser.py --list \
    > "$run/denoiser-probes.txt" 2>&1 || true
sed 's/^/   /' "$run/denoiser-probes.txt" | head -7

# A probe's name is the reference's; `case.json` is the request's own statement, so the script
# takes it rather than inventing one - a case this script made up would compare against a request
# nobody serves. `artifacts/diagnosis/denoiser-rows/case.json` is a stated example.
case_file=${H3_DIAGNOSIS_CASE:-}
if [ -n "$case_file" ] && [ -f "$case_file" ]; then
    for probe in noise rows; do
        $reference_python crates/models/minimax-h3/tools/dump_denoiser.py --checkpoint "$root" \
            --case "$case_file" --probe "$probe" --out "$run/ref-$probe" \
            > "$run/ref-$probe.log" 2>&1 || true
        if [ -d "$run/ref-$probe" ] && [ -n "$(ls -A "$run/ref-$probe" 2>/dev/null)" ]; then
            echo "   ${probe}: the reference's dump is $run/ref-$probe"
        else
            # A probe that cannot run must say what it was missing; an empty directory reads as a
            # probe that ran and found nothing.
            missing=$(grep -m1 -E "Error|OSError|ModuleNotFound|No module" "$run/ref-$probe.log" 2>/dev/null \
                | sed 's/^ *//')
            echo "   ${probe}: not drawn here - ${missing:-no reason printed in $run/ref-$probe.log}"
        fi
    done
else
    echo "   the host-only probes did not run: H3_DIAGNOSIS_CASE names no case file"
    echo "   (a stated example is artifacts/diagnosis/denoiser-rows/case.json)"
fi

cat >"$run/denoiser-commands.txt" <<COMMANDS
The denoiser's blocks (stage 3's subject, no canon number of their own) numeric half: the
reference's probes, then ours against them.

  # the reference's host-only probes, on the host that can import its stack
  $reference_python crates/models/minimax-h3/tools/dump_denoiser.py --checkpoint $root \\
      --case <case.json> --probe noise --out $run/ref-noise
  $reference_python crates/models/minimax-h3/tools/dump_denoiser.py --checkpoint $root \\
      --case <case.json> --probe rows --out $run/ref-rows

  # ours against each of them, as a per-block report
  cargo run -p minimax-h3 --example compare_blocks -- --dumps $run/ref-noise \\
      --candidate <candidate dump dir> --out $run/report-noise.md

  # and the served forward's own trajectory, which a runner dumps under \$UNISERVE_DIAG_DIR
  $reference_python crates/models/minimax-h3/tools/dump_denoiser.py --probe trajectory \\
      --checkpoint $root --case <case.json> --out $run/ref-trajectory
  python3 tools/diagnosis/compare.py --ref $run/ref-trajectory --got <diag dir> \\
      --out $run/report-denoiser.json
COMMANDS
sed 's/^/   /' "$run/denoiser-commands.txt" | head -3


# 2c. Stages 4 and 5: the video encoder, and the conditioned path that binds it. The comparison is
#     over the rows the encoder emits - `rows.h3t`, `[rows, channels * 4]` FP32 - which the stage
#     writes beside the pixels it read, and which the reference's own encode probe computes from the
#     same pixels. Comparing the projected prefix instead would attribute the projection's rounding
#     to the encoder.
echo
echo "== stages 4/5: the video encoder's rows"
# The prefixes are the stage's own: `fl2va places …` names the blocks, and the two `… encoder:`
# lines are each key list against its component's index. A grep for prefixes no stage prints would
# print nothing and read as a stage that passed.
cargo run -q -p minimax-h3 --example conditioned -- --stage weights --root "$root" 2>&1 \
    | grep -E '^(fl2va|t2va|ref2va|video encoder|audio encoder|audio_vae)' \
    | sed 's/^/   /' || true

cat >"$run/stage45-commands.txt" <<COMMANDS
Stages 4 and 5 numeric half: the conditioned stage writes the encoder's unprojected rows, and the
reference's probe compares them. Both run on the host that has the shards and a device.

  # ours: one image, the encoder's tile walk, the posterior, the anchor and the projection
  cargo run --release -p minimax-h3 --example conditioned -- --stage encode \\
      --root $root --store <store> --out $run/stage5 --canvas 256x256 \\
      --image 256 --frames 5 --text-tokens 15 --seed 0

  # the reference on the same pixels, compared against $run/stage5/latent_encoder/rows.h3t
  $reference_python crates/models/minimax-h3/tools/dump_vae_blocks.py --probe encode_rows \\
      --checkpoint $root --stack $run/stage5/latent_encoder --out $run/stage5

The probe prints the relative L2 and the maximum absolute difference, or names the shape
disagreement that stops the two rows being comparable. A case that carries no rows.h3t gets the
reference side alone and the page says so - that is a stage that did not run, not an agreement.

Recorded here rather than run because this host holds the indexes and none of the shards: the
weights stage above answers the names, and the first payload read stops.
COMMANDS
sed 's/^/   /' "$run/stage45-commands.txt" | head -3

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
    if [ -f "$run/report-noise.md" ]; then
        echo "   the denoiser's blocks have a comparison: $run/report-noise.md"
    else
        echo "   the denoiser's blocks have no numbers on this host: see $run/denoiser-commands.txt"
    fi
} | tee "$run/reading.txt"

echo
echo "artifacts under $run: the schedule dump, its comparison, the probe catalogue, the stage-1"
echo "and denoiser commands, and the reading."
echo "This is a diagnosis (design 10): nothing above gates the branch."
