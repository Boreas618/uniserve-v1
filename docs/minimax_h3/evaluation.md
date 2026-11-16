# Evaluation

The evaluator, `uniserve-eval` (`tools/eval`), runs fixed workloads against the deployments in the [guide](minimax_h3.md) and validates every MP4 against the shape its request states: size, frame count, frame rate and stereo soundtrack, and for UniServe also the shape the server reports generating. Each run writes its resolved commands, request data, per-request latencies, media validation and GPU storage observations into one bundle per point. The evaluator knows no model and no campaign: a run is given a profile, a TOML file stating the servers it starts and the points it measures, which is written at use from the protocol below. Resolve a profile with `plan` before running it, and run points serially, one server and one harness at a time.

The protocol's inputs lie beside this page, under `evaluation/`: the request manifests (`workloads/`), the source of the fixed-length prompt (`fixed-length.json`) and the FastVideo baseline's server configuration (`fastvideo-omniref.yaml`). Each manifest row is one request; a referenced row's `metadata.condition_media` gives every reference file's path under the inputs directory, its source URL and its SHA-256.

## Writing a run

The shell functions below write the parts of a profile; every campaign of this page composes its profile from them. Each function prints TOML; a campaign's profile is the concatenation of its servers and points.

```bash
cargo build --release -p eval
EVAL=$PWD/target/release/uniserve-eval
E=$PWD/docs/minimax_h3/evaluation
UNISERVE=$(command -v uniserve)            # the executable every UniServe server runs
INPUTS=/workspace/uniserve/artifacts/minimax_h3/inputs   # the referenced workloads' media

# words WORD...: the words as a TOML array of literal strings.
words() {
  local list= word
  for word; do list+="${list:+, }'$word'"; done
  printf '[%s]' "$list"
}

# server NAME [VARIABLE=VALUE...] -- COMMAND...: a server on this host's port 18083, ready once
# /health answers, run with the environment variables given.
server() {
  local name=$1 environment=
  shift
  while [ "$1" != -- ]; do environment+="${environment:+, }${1%%=*} = '${1#*=}'"; shift; done
  shift
  printf '[servers.%s]\nhost = "127.0.0.1"\nport = 18083\nready_path = "/health"\ncommand = %s\nenvironment = { %s }\n\n' \
    "$name" "$(words "$@")" "$environment"
}

# uniserve NAME MODEL SERVED CONFIG [OPTION...]: a UniServe server of the checkpoint MODEL under
# the name SERVED, on the deployment file configs/minimax_h3/CONFIG.toml.
uniserve() {
  local name=$1 model=$2 served=$3 config=$4
  shift 4
  server "$name" -- "$UNISERVE" serve --model-dir "$model" --served-model-name "$served" \
    --host 127.0.0.1 --port 18083 --config "$PWD/configs/minimax_h3/$config.toml" "$@"
}

# video NAME SERVER SERVED WORKLOAD MEASURED [FIELD...]: a point that sends the rows of
# workloads/WORKLOAD-warmup.jsonl unmeasured, then the first MEASURED rows of WORKLOAD.jsonl one at
# a time, to /v1/videos/sync of SERVER under the model name SERVED, each answer validated as the
# 1344x768, 24-fps MP4 with a stereo 32-kHz soundtrack every workload of this page states. A FIELD
# is `load.KEY=VALUE`, `video.KEY=VALUE` or `KEY=VALUE` (TOML values), which adds that field to the
# point's load, its video table or the point itself: `load.max_concurrency` (1 unless given),
# `load.request_timeout_s` (3600 unless given), `endpoint`.
video() {
  local name=$1 server=$2 served=$3 workload=$4 measured=$5 load= video= top= field
  shift 5
  for field; do
    case $field in
      load.*) load+=", ${field#load.}" ;;
      video.*) video+=", ${field#video.}" ;;
      *) top+="${field%%=*} = ${field#*=}"$'\n' ;;
    esac
  done
  [[ $load == *max_concurrency* ]] || load+=", max_concurrency = 1"
  [[ $load == *request_timeout_s* ]] || load+=", request_timeout_s = 3600"
  [[ $top == *endpoint* ]] || top+='endpoint = "/v1/videos/sync"'$'\n'
  cat <<EOF
[benchmarks.$name]
server = "$server"
task = "video"
model = "$served"
dataset = "jsonl"
dataset_path = "$E/workloads/$workload.jsonl"
${top}video = { size = "1344x768", fps = 24, audio_sample_rate = 32000, audio_channels = 2$video }
load = { num_prompts = $measured, request_rate = inf, warmup_manifest = "$E/workloads/$workload-warmup.jsonl"$load }
metrics = { videos_per_second = "higher", "video_latency_ms.p50" = "lower", "video_latency_ms.p95" = "lower" }

EOF
}

# suite NAME POINT...: the points one `run NAME` measures, in order.
suite() {
  local name=$1
  shift
  printf '[suites.%s]\npoints = %s\n\n' "$name" "$(words "$@" | tr "'" '"')"
}
```

`plan` prints the points a selection resolves to, with every server command and environment as it will run; `run` measures them into `<output root>/<point>`, and `--reuse-deployment` keeps one server for consecutive points of the same server. A two-host server starts the leader on `rank-0` with `--host-identity rank-0 --nnodes 2 --dist-init-addr $DIST` (`DIST` is `rank-0`'s address and a free port); start the daemon of `rank-1` against the same address for each server the run starts, as in [Two hosts](minimax_h3.md#two-hosts).

## FastH3 post

The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) measures FastH3 8-Step V2 and its NVFP4 export. The workload is 72 latency requests one at a time, 12 for each combination of a 5, 10 or 15 second clip and a 1000- or 10000-token prompt (`fasth3-latency`), and 32 throughput requests at each concurrency after the priming requests of that concurrency (`fasth3-throughput`, `fasth3-priming-c<N>`), all preceded by six unmeasured warmup requests (`fasth3-warmup`). Each hardware configuration serves a latency deployment and a throughput deployment of each checkpoint:

| Configuration | Latency deployment | Throughput deployment | Concurrencies | Memory fraction |
| --- | --- | --- | --- | --- |
| `gb200-4` | `ulysses4`, 2 running | the latency deployment | 1, 2, 4, 8 | 0.70 |
| `gb200-8` | `ulysses8-two-host`, 2 running | `fasth3-ulysses4x2-two-host`, 4 running | 1, 2, 4, 8, 16 | 0.70 |
| `rtx-pro-6000-8` | `ulysses8`, 2 running, `text_capacities=[1024, 10240]` | `fasth3-ulysses4x2`, 4 running, the same capacities | 1, 2, 4, 8, 16 | 0.92 |

Every FastH3 deployment states `--max-model-len 16384 --set video_config.max_seconds=15 --set 'dtype="bfloat16"'`; the BF16 checkpoint adds `--set 'quantization.mode="quality"'`, and the NVFP4 export loads the representation its own configuration states. The `gb200-8` throughput deployment is one Ulysses4 replica on each host; its latency deployment places one Ulysses8 replica across both hosts, whose products the hosts' Storage Servers copy between them.

```bash
BF16=/workspace/models/FastVideo-FastH3-8-Step-V2
NVFP4=/workspace/models/FastVideo-FastH3-8-Step-V2-NVFP4
TWO_HOST=(--host-identity rank-0 --nnodes 2 --dist-init-addr "$DIST")
FASTH3=(--max-model-len 16384 --set video_config.max_seconds=15 --set 'dtype="bfloat16"')

# fasth3 CONFIGURATION PRECISION LATENCY THROUGHPUT CONCURRENCIES: the servers and points of one
# configuration and precision (bf16, nvfp4), its LATENCY and THROUGHPUT servers named, and the suite
# of its latency point and its throughput points at each of CONCURRENCIES.
fasth3() {
  local configuration=$1 precision=$2 latency=$3 throughput=$4 concurrencies=$5 points= c
  local name=$configuration-$precision
  video "$name-latency" "$latency" FastH3 fasth3-latency 72 load.request_timeout_s=21600
  points+=" $name-latency"
  for c in $concurrencies; do
    video "$name-throughput-c$c" "$throughput" FastH3 fasth3-throughput 32 load.max_concurrency=$c \
      load.request_timeout_s=21600 "load.priming_manifest=\"$E/workloads/fasth3-priming-c$c.jsonl\""
    points+=" $name-throughput-c$c"
  done
  suite "$name" $points
}

for precision in bf16 nvfp4; do
  model=$BF16 quality=(--set 'quantization.mode="quality"')
  [ $precision = nvfp4 ] && model=$NVFP4 quality=()
  {
    uniserve gb200-4-$precision $model FastH3 ulysses4 --max-running-requests 2 "${FASTH3[@]}" --mem-fraction-static 0.70 "${quality[@]}"
    fasth3 gb200-4 $precision gb200-4-$precision gb200-4-$precision "1 2 4 8"
  } > artifacts/minimax_h3/fasth3-gb200-4-$precision.toml
  {
    uniserve gb200-8-$precision-latency $model FastH3 ulysses8-two-host --max-running-requests 2 "${FASTH3[@]}" --mem-fraction-static 0.70 "${TWO_HOST[@]}" "${quality[@]}"
    uniserve gb200-8-$precision-throughput $model FastH3 fasth3-ulysses4x2-two-host --max-running-requests 4 "${FASTH3[@]}" --mem-fraction-static 0.70 "${TWO_HOST[@]}" "${quality[@]}"
    fasth3 gb200-8 $precision gb200-8-$precision-latency gb200-8-$precision-throughput "1 2 4 8 16"
  } > artifacts/minimax_h3/fasth3-gb200-8-$precision.toml
  {
    uniserve rtx-pro-6000-8-$precision-latency $model FastH3 ulysses8 --max-running-requests 2 "${FASTH3[@]}" --mem-fraction-static 0.92 --set 'video_config.text_capacities=[1024, 10240]' "${quality[@]}"
    uniserve rtx-pro-6000-8-$precision-throughput $model FastH3 fasth3-ulysses4x2 --max-running-requests 4 "${FASTH3[@]}" --mem-fraction-static 0.92 --set 'video_config.text_capacities=[1024, 10240]' "${quality[@]}"
    fasth3 rtx-pro-6000-8 $precision rtx-pro-6000-8-$precision-latency rtx-pro-6000-8-$precision-throughput "1 2 4 8 16"
  } > artifacts/minimax_h3/fasth3-rtx-pro-6000-8-$precision.toml
done

$EVAL plan --config artifacts/minimax_h3/fasth3-gb200-4-bf16.toml gb200-4-bf16
$EVAL run --config artifacts/minimax_h3/fasth3-gb200-4-bf16.toml gb200-4-bf16 --output-root artifacts/minimax_h3/fasth3
```

The W1 and W2 workloads of [MiniMax-H3 workloads](#minimax-h3-workloads) run on FastH3 8-Step V2 on the `gb200-4` BF16 deployment as `video gb200-4-bf16-w1 gb200-4-bf16 FastH3 w1 8` and `video gb200-4-bf16-w2 gb200-4-bf16 FastH3 w2 4`.

## H200

These use `FastVideo/FastVideo-FastH3-8-Step-V2` at revision `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, the fixed-length prompt of `evaluation/fixed-length.json` at the stated token count, every request at seed 1000, full graphs, `--mem-fraction-static 0.92` and `NCCL_CUMEM_HOST_ENABLE=1`. Latency is the mean ± SD of three requests at concurrency one after one warmup; the DP8 points use eight warmups and 16 requests at concurrency eight.

| Deployment | Precision | Workload | Result |
| --- | --- | --- | --- |
| Eight-H200 Ulysses | `quality` | 5 s, 1000 tokens | 8.194 ± 0.818 s |
| Eight-H200 Ulysses | `quality` | 5 s, 10000 tokens | 14.218 ± 0.880 s |
| Eight-H200 Ulysses | `quality` | 10 s, 1000 tokens | 18.539 ± 1.071 s |
| Eight-H200 Ulysses | `quality` | 10 s, 10000 tokens | 26.578 ± 0.600 s |
| Eight-H200 Ulysses | `quality` | 15 s, 1000 tokens | 31.069 ± 0.045 s |
| Eight-H200 Ulysses | `quality` | 15 s, 10000 tokens | 44.228 ± 0.878 s |
| Four-H200 Ulysses | `quality` | 5 s, 1000 tokens | 13.864 ± 0.013 s |
| Gather8 | `quality` | 5 s, 10000 tokens | 20.170 ± 0.698 s |
| Eight-H200 Ulysses | FP8 MLPs | 5 s, 1000 tokens | 7.448 ± 0.734 s |
| Eight-H200 Ulysses | FP8 MLPs | 15 s, 10000 tokens | 42.380 ± 0.807 s |
| DP8 + text TP8 | `quality` | 5 s, 1000 tokens, concurrency 8 | 0.1519 videos/s, 52.231 ± 0.358 s, 90,329 MiB peak per GPU |
| DP8 + text TP8 | FP8 MLPs | 5 s, 1000 tokens, concurrency 8 | 0.1676 videos/s, 47.314 ± 0.386 s, 81,531 MiB peak per GPU |

Each row is one point: a request of the stated length (`num_frames` 124, 243 or 362 for 5, 10 or 15 s) and prompt (`prompt_tokens` 1000 or 10000), three measured after one warmup at concurrency one, and 16 after eight at concurrency eight for the DP8 rows. The deployments state `--max-model-len 10240 --set 'video_config.text_capacities=[1024]' --set video_config.max_seconds=15 --max-running-requests 2` (the Ulysses8 and Gather8 rows) or `--max-model-len 1024 --set video_config.max_seconds=5` with 2 running for `ulysses4` and 8 for the DP8 deployment `fasth3-dp8-text-tp8`; the `quality` rows state `--set 'quantization.mode="quality"'` and the FP8 rows `--set 'quantization={ modules = { mlp = "fp8" } }'`. Every H200 server runs with `CUDA_VISIBLE_DEVICES` naming its GPUs, `OMP_NUM_THREADS=1`, `MAX_JOBS=8` and `NCCL_CUMEM_HOST_ENABLE=1`:

```bash
H200=(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MAX_JOBS=8 NCCL_CUMEM_HOST_ENABLE=1)

# prompted NAME SERVER FRAMES TOKENS MEASURED CONCURRENCY WARMUPS: a point of WARMUPS unmeasured and
# MEASURED measured requests of the fixed-length prompt of TOKENS tokens, for FRAMES frames.
prompted() {
  cat <<EOF
[benchmarks.$1]
server = "$2"
task = "video"
model = "FastH3"
dataset = "fixed-length"
dataset_path = "$E/fixed-length.json"
tokenizer = "$BF16/tokenizer"
video = { size = "1344x768", num_frames = $3, fps = 24, audio_sample_rate = 32000, audio_channels = 2, prompt_tokens = $4 }
load = { num_prompts = $5, request_rate = inf, max_concurrency = $6, warmup_requests = $7 }
metrics = { videos_per_second = "higher", "video_latency_ms.mean" = "lower" }

EOF
}

{
  server h200-ulysses8 "${H200[@]}" -- "$UNISERVE" serve --model-dir $BF16 --served-model-name FastH3 \
    --config $PWD/configs/minimax_h3/ulysses8.toml --host 127.0.0.1 --port 18083 --max-model-len 10240 \
    --set 'video_config.text_capacities=[1024]' --set video_config.max_seconds=15 --max-running-requests 2 \
    --mem-fraction-static 0.92 --set 'quantization.mode="quality"'
  points=
  for frames in 124:5s 243:10s 362:15s; do
    for tokens in 1000:1k 10000:10k; do
      prompted h200-ulysses8-${frames#*:}-${tokens#*:} h200-ulysses8 ${frames%%:*} ${tokens%%:*} 3 1 1
      points+=" h200-ulysses8-${frames#*:}-${tokens#*:}"
    done
  done
  suite h200-ulysses8 $points
} > artifacts/minimax_h3/fasth3-h200-ulysses8.toml
$EVAL run --config artifacts/minimax_h3/fasth3-h200-ulysses8.toml h200-ulysses8 --reuse-deployment --output-root artifacts/minimax_h3/fasth3-h200
```

The other rows compose the same way: `ulysses4` on `CUDA_VISIBLE_DEVICES=0,1,2,3` for its 5 s, 1000-token row; `gather8` for its 5 s, 10000-token row; the FP8 `ulysses8` deployment for its two rows; and the DP8 deployments, `quality` and FP8, each for its 5 s, 1000-token row at `prompted NAME SERVER 124 1000 16 8 8`.

## RTX PRO 6000

Three deployments run the throughput points of the [FastH3 post](#fasth3-post) on one host with eight RTX PRO 6000 Blackwell Server Edition GPUs: six unmeasured warmups, the priming requests of the concurrency, then the 32 throughput requests, at concurrency 8, and 16 on the TP8 text encoders. Each serves eight one-GPU denoiser replicas, which keep every denoiser collective off PCIe: the NVFP4 checkpoint beside one TP8 text encoder (`fasth3-dp8-text-tp8.toml`) or two TP4 text encoders, one per four-GPU half (`fasth3-dp8-text-tp4x2.toml`), and the BF16 checkpoint with each replica's layer projections in host storage (`fasth3-dp8-text-tp8-host-weights.toml`). A one-GPU replica holds a whole 15-second sequence, so the DP8 servers bound prompts at 10240 tokens (the workload's longest is 10000) and prepare only the 16:9 canvas. Collectives use NCCL's default transport: on a host whose GPUs share no PCIe switch, `NCCL_P2P_LEVEL=SYS` makes NCCL use PCIe peer-to-peer between them where its default connects them through host storage, which lowers its all-to-all and all-reduce bandwidth.

They state `--max-running-requests 8 --max-model-len 10240 --set video_config.max_seconds=15 --set 'dtype="bfloat16"' --mem-fraction-static 0.92 --set 'video_config.text_capacities=[1024]' --set 'video_config.sizes=["1344x768"]'`, the BF16 checkpoint adding `--set 'quantization.mode="quality"'`, and each measures its throughput points at concurrency 8 and, beside a TP8 text encoder, 16:

```bash
{
  uniserve rtx-pro-6000-8-dp8-nvfp4 $NVFP4 FastH3 fasth3-dp8-text-tp8 --max-running-requests 8 --max-model-len 10240 \
    --set video_config.max_seconds=15 --set 'dtype="bfloat16"' --mem-fraction-static 0.92 \
    --set 'video_config.text_capacities=[1024]' --set 'video_config.sizes=["1344x768"]'
  for c in 8 16; do
    video rtx-pro-6000-8-dp8-nvfp4-throughput-c$c rtx-pro-6000-8-dp8-nvfp4 FastH3 fasth3-throughput 32 \
      load.max_concurrency=$c load.request_timeout_s=21600 "load.priming_manifest=\"$E/workloads/fasth3-priming-c$c.jsonl\""
  done
  suite rtx-pro-6000-8-dp8-nvfp4 rtx-pro-6000-8-dp8-nvfp4-throughput-c8 rtx-pro-6000-8-dp8-nvfp4-throughput-c16
} > artifacts/minimax_h3/fasth3-rtx-pro-6000-8-dp8-nvfp4.toml
$EVAL run --config artifacts/minimax_h3/fasth3-rtx-pro-6000-8-dp8-nvfp4.toml rtx-pro-6000-8-dp8-nvfp4 --reuse-deployment --output-root artifacts/minimax_h3/fasth3-rtx
```

The two-TP4 deployment (`fasth3-dp8-text-tp4x2`) measures concurrency 8 only; the BF16 deployment (`fasth3-dp8-text-tp8-host-weights`) measures 8 and 16.

Twelve cells of the OmniRef reference matrix, the `image`, `image-voice`, `video` and `mixed` compositions at 5, 10 and 15 seconds, run on one Ulysses8 reference replica (`ulysses8-reference.toml`) with the capacities, sizing and seeds of the [OmniRef matrix on H200](#omniref-on-h200) and graphs off; SM120 evaluates each 128-row sparse-attention tile as two tiles of its 64-row kernel. A smoke slice runs five of those cells: the four compositions at 5 s and `image` at 15 s.

No measurement of these configurations is recorded.

## MiniMax-H3 workloads

These workloads run on UniServe, SGLang, vLLM-Omni and FastVideo:

| Workload | Request | Output | Measured seeds |
| --- | --- | --- | --- |
| W1 | t2va, official prompt | 1344×768, 124 frames | 0–7 |
| W2 | t2va, official prompt | 1344×768, 362 frames | 0–3 |
| W3 | fl2va, official first keyframe (1920×1080) | 1344×768, 192 frames | 0–7 |
| W4 | ref2va, reference image and voice | 1344×768, 124 frames | 0–7 |
| W5 | ref2va, reference video with its soundtrack and a reference voice | 1344×768, 124 frames | 0–7 |

Every output is 24 fps with a stereo 32-kHz soundtrack. A point states the output's `size`, `fps`, `audio_sample_rate` and `audio_channels` in its `video` table, and each manifest row states its `num_frames`, which must be a count the model generates; UniServe receives exactly that shape in its `/v1/videos` body and derives the task from the row's `input_references`. A baseline point also states the requested target its native MiniMax-H3 form names: SGLang and vLLM-Omni an `aspect_ratio` (`16:9`, or `auto` where the canvas follows the keyframe or the default) and a `duration_seconds` (5, 8 or 15) that resolve to the stated shape, FastVideo the `duration_seconds`.

Each point sends one unmeasured warmup at seed 42, then its measured requests one at a time: 8 for W1, W3, W4 and W5 and 4 for W2. A referenced workload's point states the inputs directory as `video.reference_root`, and its UniServe server reads media from it (`--set media_config.directory="$INPUTS"`). Reference media decode through the FFmpeg 8.1 libraries the `uniserve` executable links. The UniServe deployments of MiniMax-H3 (`/workspace/models/MiniMax-H3`) and of FastH3 OmniRef (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`, with the base revision it pins in the Hugging Face cache) are:

| Server | Checkpoint | Deployment file | Options beyond `--max-running-requests 2 --mem-fraction-static 0.70 --set 'dtype="bfloat16"' --set 'quantization.mode="quality"'` | Workloads |
| --- | --- | --- | --- | --- |
| `gb200-4` | MiniMax-H3 | `ulysses4` | `--max-model-len 1024 --set 'video_config.text_capacities=[1024]' --set video_config.max_seconds=15` | W1, W2 |
| `gb200-4-keyframe` | MiniMax-H3 | `ulysses4` | `--max-model-len 2048 --set 'video_config.text_capacities=[2048]' --set video_config.max_seconds=8`, the media directory | W3 |
| `gb200-4-reference` | MiniMax-H3 | `ulysses4-reference` | `--max-model-len 8192 --set 'video_config.text_capacities=[8192]' --set video_config.max_seconds=5 --set video_config.max_condition_rows=40960 --set mm_processor_kwargs.enlarge_references=true`, the media directory | W4, W5 |
| `gb200-4-omniref` | FastH3 OmniRef | `ulysses4-reference` | as `gb200-4-reference`, with `--set video_config.max_condition_rows=49152` | W4, W5 |
| `gb200-8` | MiniMax-H3 | `ulysses8-two-host` | as `gb200-4`, on two hosts | W1, W2 |
| `gb200-8-keyframe` | MiniMax-H3 | `ulysses8-two-host` | as `gb200-4-keyframe`, on two hosts | W3 |

The reference servers enlarge references to their encoding size (`enlarge_references`), the sizing of the released pipeline and of the SGLang and FastVideo baselines W4 and W5 are compared with. For example, W3 on four GB200s:

```bash
H3=/workspace/models/MiniMax-H3
{
  uniserve gb200-4-keyframe $H3 MiniMax-H3 ulysses4 --max-running-requests 2 --max-model-len 2048 \
    --set 'video_config.text_capacities=[2048]' --set video_config.max_seconds=8 \
    --set "media_config.directory=\"$INPUTS\"" --mem-fraction-static 0.70 --set 'dtype="bfloat16"' \
    --set 'quantization.mode="quality"'
  video gb200-4-w3 gb200-4-keyframe MiniMax-H3 w3 8 "video.reference_root=\"$INPUTS\""
} > artifacts/minimax_h3/w3-gb200-4.toml
$EVAL plan --config artifacts/minimax_h3/w3-gb200-4.toml gb200-4-w3
$EVAL run --config artifacts/minimax_h3/w3-gb200-4.toml gb200-4-w3 --output-root artifacts/minimax_h3/workloads
```

A baseline point names its engine's request form in `video.backend` and states the target its native MiniMax-H3 form names, with the base schedule:

| Engine | Server | Point fields |
| --- | --- | --- |
| SGLang | `$SGLANG/bin/sglang serve --model-path $H3 --model-variant fl2va` (W1–W3) or `ref2va` (W4, W5) `--num-gpus 4 --ulysses-degree 4 --performance-mode speed --disable-conditioning-cache --output-path $SCRATCH/sglang-minimax-h3/outputs --input-save-path $SCRATCH/sglang-minimax-h3/inputs --host 127.0.0.1 --port 18083`, with `PATH=$SGLANG/bin:$CUDA_HOME/bin:$PATH`, `SGLANG_RUST_BUILD_MODE=never` and `EP_SUPPRESS_NCCL_CHECK=1` | `endpoint="/v1/videos"`, `video.backend="sglang"`, `video.num_inference_steps=50`, `video.flow_shift=12.0`, `video.audio_flow_shift=3.0`, `video.extra_params={ quality = "lossless" }` |
| vLLM-Omni | `$VLLM_OMNI/bin/vllm serve --model-dir $H3/FL2VA` (W1–W3) or `$H3/Ref2VA` with `--text-encoder-tp-size 4` (W4, W5) `--omni --served-model-name MiniMax-H3 --host 127.0.0.1 --port 18083 --trust-remote-code --task-type fl2va` or `ref2va` `--num-gpus 4 --usp 4 --ring 1 --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling --cache-backend none`, with `PATH=$VLLM_OMNI/bin:$PATH`, `LD_LIBRARY_PATH=$VLLM_OMNI/opt/x11-lib`, `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600` | `video.backend="vllm-omni"`, `video.num_inference_steps=50`, `video.flow_shift=12.0`, `video.audio_flow_shift=3.0` |
| FastVideo | `$FASTVIDEO/bin/fastvideo serve --config $E/fastvideo-omniref.yaml --generator.model_path $FASTVIDEO_OMNIREF`, with `PATH=$FASTVIDEO/bin:$PATH` and `HF_HUB_OFFLINE=1` (W4, W5) | `video.backend="fastvideo"`, `video.num_inference_steps=9`, `video.parallel_decoding=true` |

Each baseline point also states `video.duration_seconds`: 5 for W1, W4 and W5, 15 for W2 and 8 for W3; SGLang and vLLM-Omni also `video.aspect_ratio`, `"16:9"` for W1 and W2 and `"auto"` for W3, W4 and W5. The two-host SGLang server serves W1–W3 with `--num-gpus 8 --sp-degree 8 --ulysses-degree 4 --ring-degree 2 --nnodes 2 --node-rank 0 --dist-init-addr $HEAD:20000 --encoder-parallel replicate`, where `HEAD` is `rank-0`'s address and `SCRATCH` a directory both hosts mount; start SGLang's node rank 1 on `rank-1`.

Cross-engine results are not equal-work ratios: the engines differ in reference-image resizing, noise generators, media codecs and distributed attention. vLLM-Omni's W4 image processing keeps a smaller condition layout than the others, and its diffusion worker does not implement the two-host rank layout.

## OmniRef reference compositions

The reference matrix runs FastH3 OmniRef (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`) on nine reference compositions of MiniMax-H3's ref2va interface, each at 124, 243 and 362 frames (5, 10 and 15 seconds at 24 fps), every output 1344×768. A workload is `ref-<composition>-<seconds>s`:

| Composition | References | Prompt tokens with vision | Condition rows | Enlarged: prompt tokens | Enlarged: condition rows |
| --- | --- | --- | --- | --- | --- |
| `image` | One 16:9 subject image | about 3,900 | 3,712 | about 7,600 | 7,296 |
| `image-voice` | One 16:9 subject image and a reference voice | about 3,950 | 4,224 to 4,480 | about 7,650 | 7,808 to 8,064 |
| `images3` | Three 16:9 images: a subject, a product sheet and a scene | about 8,900 | 8,576 | about 22,400 | 22,016 |
| `images9` | Nine square images | 14,415 | 13,952 | 37,701 | 36,864 |
| `clip` | One 2-second reference clip with its soundtrack | about 3,500 | 14,080 | about 3,500 | 14,080 |
| `video` | One reference video (6 s, with its soundtrack) | 6,458 to 7,630 | 46,592 to 51,200 | 6,458 to 7,630 | 46,592 to 51,200 |
| `clips3` | Three 2-second reference clips, each with its soundtrack | 9,718 | 42,240 | 9,718 | 42,240 |
| `mixed` | One 16:9 image, one reference video with its soundtrack and a reference voice | 10,215 to 11,422 | 50,816 to 55,680 | 13,911 to 15,118 | 54,400 to 59,264 |
| `full` | Nine square images and three 2-second clips with their soundtracks | 23,874 | 56,192 | 47,160 | 79,104 |

A request's packed sequence is its prompt, which carries every visual reference's vision tokens, its condition rows and its generated rows: 37,710 at 5 s, 73,386 at 10 s and 109,062 at 15 s. A reference video or voice longer than the target contributes the target's duration, so its rows grow with the target up to its own length. The square images are centred crops and the clips the leading two seconds of their sources, cut without re-encoding; each row's `metadata.condition_media` names the source and the derivation.

The matrix runs under both reference sizings (`enlarge_references` of `mm_processor_kwargs`). The default sizing: a reference keeps its own size unless it exceeds its encoding size. Every image of the matrix has a short edge below 2048 pixels, so each is as many prompt tokens and condition rows as its own 32-pixel patches, 3,600 for the 2560×1440 subject and 576 to 2,025 for a square, and every reference video is larger than its canvas and scales down to it; the largest cell, `ref-full-15s`, packs 189,830 rows. The enlarged sizing enlarges each image to its 2048-pixel short edge, 7,296 prompt tokens and condition rows at 16:9 and 4,096 when square, as the released pipeline and SGLang do; its largest cell packs 236,294 rows.

A point sends one warmup at seed 42 and three measured requests one at a time, seeds 0 to 2, each allowed two hours. One deployment, a Ulysses8 replica across two four-GB200 hosts (`ulysses8-reference-two-host.toml`), serves every cell of a sizing, with a text capacity for each cell's prompt and the condition rows of `full`: `--set video_config.max_seconds=15 --max-model-len 24576 --set 'video_config.text_capacities=[4096, 7168, 8192, 9216, 10240, 12288, 15360]' --set video_config.max_condition_rows=56192` by default, and `--max-model-len 48128 --set 'video_config.text_capacities=[4096, 7168, 8192, 10240, 14336, 15360, 22528, 37888]' --set video_config.max_condition_rows=79104 --set mm_processor_kwargs.enlarge_references=true` for the enlarged sizing.

```bash
OMNIREF=/workspace/models/FastVideo-FastH3-Omni-8-Step-V1
{
  uniserve gb200-8-omniref-refs $OMNIREF MiniMax-H3 ulysses8-reference-two-host --max-running-requests 2 \
    --max-model-len 24576 --set 'video_config.text_capacities=[4096, 7168, 8192, 9216, 10240, 12288, 15360]' \
    --set video_config.max_seconds=15 --set video_config.max_condition_rows=56192 \
    --set "media_config.directory=\"$INPUTS\"" --mem-fraction-static 0.70 --set 'dtype="bfloat16"' "${TWO_HOST[@]}" \
    --set 'quantization.mode="quality"'
  points=
  for seconds in 5 10 15; do
    for composition in image image-voice images3 images9 clip video clips3 mixed full; do
      video gb200-8-omniref-ref-$composition-${seconds}s gb200-8-omniref-refs MiniMax-H3 ref-$composition-${seconds}s 3 \
        load.request_timeout_s=7200 "video.reference_root=\"$INPUTS\""
      points+=" gb200-8-omniref-ref-$composition-${seconds}s"
    done
  done
  suite gb200-8-omniref-refs $points
} > artifacts/minimax_h3/omniref-gb200-8.toml
$EVAL run --config artifacts/minimax_h3/omniref-gb200-8.toml gb200-8-omniref-refs --reuse-deployment --output-root artifacts/minimax_h3/omniref
```

SGLang's two-host reference server (`--model-variant ref2va` with the two-host options of [MiniMax-H3 workloads](#minimax-h3-workloads)) measures SGLang's base Ref2VA on the same cells and hosts with the base schedule (50 sigma points), one measured request per cell after its warmup, with `video.aspect_ratio="16:9"` and the cell's `video.duration_seconds`; start SGLang's node rank 1 on `rank-1`. SGLang enlarges every reference image, so its results compare with the enlarged suite.

## OmniRef on H200

FastH3 OmniRef runs on H200 with the SM90 kernel of its 128-row sparse-attention tiles. The reference matrix runs on one eight-H200 host as one Ulysses8 replica (`ulysses8-reference.toml`) under both sizings, with the capacities, warmups and seeds of the [GB200 matrix](#omniref-reference-compositions), `--mem-fraction-static 0.92` and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, without the two-host options. No measurement of these points is recorded.

## Fixed check set

MiniMax-H3's quality is confirmed by a person watching and listening to the videos a fixed set of requests generates, not by an automatic quality gate. The set covers every combination the service offers — the `t2va`, `fl2va` and `ref2va` tasks against FastH3 DMD, FastH3 OmniRef and base MiniMax-H3 — as six cells. FastH3 DMD serves `t2va` and `fl2va`, FastH3 OmniRef serves `ref2va`, and the base release serves `t2va` and `fl2va` through its `transformer` partition and `ref2va` through its `transformer_ref` partition.

Each cell sends one unmeasured warmup at seed 42 and one measured request at seed 0, the seed-0 rows of W1, W3 and W4 of [MiniMax-H3 workloads](#minimax-h3-workloads): `t2va` with the official prompt for 124 frames, `fl2va` with the official first keyframe for 192 frames, and `ref2va` with the reference image and voice for 124 frames, each at 1344×768. The rows are frozen in `evaluation/workloads/check-*.jsonl` and their `-warmup.jsonl` companions as copies of the W1, W3 and W4 rows, so the set does not follow later changes to the performance workloads.

Every measured video is written to a fixed path under `artifacts/`, `artifacts/minimax_h3/check/<checkpoint>-<task>/<request id>.mp4`, so a confirmed cell keeps the path a person watches and no cell writes into another's directory: `<checkpoint>` is `fasth3-dmd`, `fasth3-omniref` or `base`, `<task>` is `t2va`, `fl2va` or `ref2va`, and `<request id>` is the row's ID, such as `w1-measured-seed0`. Each point's bundle, with its requests, media index and summary, is under `artifacts/minimax_h3/check/results/<point>`.

No server states a precision selector: the launch's default `dtype` is `bfloat16` and an empty
`quantization` loads the model's first preset, `quality`, so every cell measures the
representation the shipped performance profiles serve.


No server states a precision selector: the launch's default `dtype` is `bfloat16` and an empty `quantization` loads the model's first preset, `quality`, so every cell measures the representation the performance deployments serve.

The six cells run in two orders, and the two runs say different things. The baseline order runs `check-base-ref2va` first, because the reference implementation shares one Storage Server key between the two base servers on one checkpoint and a weight its loader computes has no file, so whichever of them runs second inherits the first's pages. Measured both ways: with the reference cell first, all six pass under `--reuse-deployment`; with it last, after the other five, its warmup fails with `invalid_video_output: nonzero_video_variance`, it passes alone on a fresh host, and it fails again once the earlier servers' Storage Server is alive and holds 61 GiB per GPU. The ordering is how the baseline is measured; it is not a property the delivered service may need.

The reversed order runs the reference cell last, each cell writing its MP4 under `artifacts/minimax_h3/check-reversed/`: the evaluator never replaces a different MP4 at an existing fixed path, so two runs of the same cells need two paths to both exist. It is the candidate's obligation: the delivered service must pass every cell **in any order, on retained weights**. A baseline-order run alone says the baseline still measures what it measured; a reversed run that passes says the ordering requirement is gone.

The automatic checks are the two this document defines for every video point and nothing more: MP4 validity (size, frame count, frame rate and stereo 32-kHz soundtrack against the stated shape, and the shape the server reports generating, which the `video` task validates) and the performance metrics each point protects, each checked to be finite and positive. The set adds no gate, tolerance or metric; the confirmation is the person's judgement of the six videos.

The four servers place four GPUs each, with `configs/minimax_h3/ulysses4.toml` for the two transformer partitions and `configs/minimax_h3/ulysses4-reference.toml` for the two reference partitions, and use port 18083 one server at a time. `--reuse-deployment` keeps a checkpoint loaded across its consecutive points, so the four checkpoints load once each. Each point states `video.media_dir`, its cell's fixed MP4 directory, and `provenance`, the checkpoint (`FastVideo/FastVideo-FastH3-8-Step-V2`, `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` or `MiniMaxAI/MiniMax-H3`) and the task:

```bash
TRANSFORMER=(--max-running-requests 2 --max-model-len 2048 --set 'video_config.text_capacities=[1024, 2048]' --set video_config.max_seconds=8 --set "media_config.directory=\"$INPUTS\"" --mem-fraction-static 0.70)
REFERENCE=(--max-running-requests 2 --max-model-len 8192 --set 'video_config.text_capacities=[8192]' --set video_config.max_seconds=5 --set "media_config.directory=\"$INPUTS\"" --mem-fraction-static 0.70 --set mm_processor_kwargs.enlarge_references=true)

# check ORDER SERVER CHECKPOINT TASK: the cell of CHECKPOINT's TASK on SERVER, writing its MP4 under
# artifacts/minimax_h3/ORDER.
check() {
  local cell=$2-$4 served=MiniMax-H3
  [ $2 = fasth3-dmd ] && served=FastH3
  [ $2 = base-transformer ] || [ $2 = base-reference ] && cell=base-$4
  video check-$cell-$1 $2 $served check-$4 1 "video.media_dir=\"$PWD/artifacts/minimax_h3/$1/$cell\"" \
    "video.reference_root=\"$INPUTS\"" "provenance={ checkpoint = \"$3\", task = \"$4\" }"
}

{
  uniserve fasth3-dmd $BF16 FastH3 ulysses4 "${TRANSFORMER[@]}"
  uniserve fasth3-omniref $OMNIREF MiniMax-H3 ulysses4-reference "${REFERENCE[@]}" --set video_config.max_condition_rows=49152
  uniserve base-transformer $H3 MiniMax-H3 ulysses4 "${TRANSFORMER[@]}"
  uniserve base-reference $H3 MiniMax-H3 ulysses4-reference "${REFERENCE[@]}" --set video_config.max_condition_rows=40960
  for order in check check-reversed; do
    [ $order = check ] && check $order base-reference MiniMaxAI/MiniMax-H3 ref2va
    check $order fasth3-dmd FastVideo/FastVideo-FastH3-8-Step-V2 t2va
    check $order fasth3-dmd FastVideo/FastVideo-FastH3-8-Step-V2 fl2va
    check $order fasth3-omniref FastVideo/FastVideo-FastH3-Omni-8-Step-V1 ref2va
    check $order base-transformer MiniMaxAI/MiniMax-H3 t2va
    check $order base-transformer MiniMaxAI/MiniMax-H3 fl2va
    [ $order = check-reversed ] && check $order base-reference MiniMaxAI/MiniMax-H3 ref2va
  done
  suite check-h3 check-base-ref2va-check check-fasth3-dmd-t2va-check check-fasth3-dmd-fl2va-check \
    check-fasth3-omniref-ref2va-check check-base-t2va-check check-base-fl2va-check
  suite check-h3-reversed check-fasth3-dmd-t2va-check-reversed check-fasth3-dmd-fl2va-check-reversed \
    check-fasth3-omniref-ref2va-check-reversed check-base-t2va-check-reversed check-base-fl2va-check-reversed \
    check-base-ref2va-check-reversed
} > artifacts/minimax_h3/check.toml
$EVAL plan --config artifacts/minimax_h3/check.toml check-h3
$EVAL run --config artifacts/minimax_h3/check.toml check-h3 --reuse-deployment --output-root artifacts/minimax_h3/check/results
$EVAL run --config artifacts/minimax_h3/check.toml check-h3-reversed --reuse-deployment --output-root artifacts/minimax_h3/check-reversed/results
```

Every measured video is written to a fixed path, `artifacts/minimax_h3/<order>/<checkpoint>-<task>/<request id>.mp4`, so a confirmed cell keeps the path a person watches and no cell writes into another's directory: `<checkpoint>` is `fasth3-dmd`, `fasth3-omniref` or `base`, `<task>` is `t2va`, `fl2va` or `ref2va`, and `<request id>` is the row's ID, such as `w1-measured-seed0`. A run on a changed service moves `artifacts/minimax_h3/check/` aside first: the evaluator refuses a non-empty result directory and never replaces a different MP4 at an existing fixed path.

A person reviews each cell from its bundle: `media_index.jsonl` names the MP4 each request wrote, and the row of its manifest states the task, the prompt, the seed and the input files the request references, with the frame each keyframe anchors. A candidate's cell is reviewed beside the baseline's of the same row ID.
