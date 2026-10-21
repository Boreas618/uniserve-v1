# UniServe

UniServe serves FastH3 8-Step text-to-video-with-audio and keyframe-conditioned generation on NVIDIA Blackwell and Hopper GPUs. Each request returns a finished MP4: 1344×768 H.264 video at 24 fps with stereo 32-kHz AAC audio. The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) describes the design and its measurements, and the [FastH3 and MiniMax-H3 guide](docs/minimax_h3/minimax_h3.md) covers every deployment, precision and option. The same guide covers the base MiniMax-H3 checkpoint's text-to-video, keyframe and reference requests and FastH3 OmniRef reference requests.

UniServe is a Rust serving system with a Python numerical library. Rust owns HTTP admission, scheduling, worker processes, request state, storage, CUDA streams and graph policy, transfers and response assembly. `uniserve` supplies the numerical layers, loading, resource binding and the models; each worker is a `uniserve worker` process that runs the Rust worker runtime and calls those same numerical implementations through TVM-FFI.

The configured model descriptions are `qwen3`, `sensenova`, `bagel`, `minimax-h3`, and `diffusion-gemma`. A server process loads exactly one description and exposes one served-model identity.

## Requirements

| Component | Requirement |
| --- | --- |
| OS / architecture | Linux on x86-64 or aarch64 |
| GPU | NVIDIA Hopper or Blackwell GPUs; H200, GB200 and RTX PRO 6000 Blackwell Server Edition have end-to-end validation |
| CUDA | CUDA 13 toolkit and a matching driver |
| Python | Python 3.12 |
| Rust | Stable Rust toolchain with edition 2024 support |

## Installation

Run from the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv sync --locked --python /usr/bin/python3.12 --extra gpu
source .venv/bin/activate
```

The sync builds the `uniserve` executable (its `serve`, `worker` and `storage` commands), the worker runtime library, and UniServe's native kernels, and installs the `gpu` extra's providers: FlashInfer and FlashAttention-4. Compiling the native kernels needs a CUDA toolkit compatible with PyTorch and a C++ compiler; they target the visible GPUs, or the architectures `TORCH_CUDA_ARCH_LIST` names. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#install) lists the system packages and a container build.

## Quickstart

Download the checkpoint and serve it on four GPUs:

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir "$H3_MODEL"

uniserve serve "$H3_MODEL" \
  --worker-groups configs/minimax_h3/ulysses4.toml \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

Startup prepares and captures every admitted request shape before `/health` reports ready; on four GB200 GPUs this takes about 11 minutes. The first startup on a machine also compiles and caches the kernels FastH3 runs and takes about 19 minutes. Until preparation finishes, the log reports `worker still busy during Worker startup` with the elapsed seconds. Then generate a video:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","task":"t2va","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5},"seed":1000}' \
  --output forest.mp4
```

## Deployments

A deployment file passed to `--worker-groups` places FastH3's components on devices and hosts: TOML `[[worker_groups]]` groups, each with its ranks (`cuda:0-3`), replicas and components. `configs/minimax_h3/` holds them; the FastH3 deployments with published measurements are:

| File | GPUs | Layout |
| --- | --- | --- |
| `ulysses4.toml` | Four on one host | Ulysses4 denoiser, TP4 text encoder |
| `ulysses8.toml` | Eight on one host | Ulysses8 denoiser, TP8 text encoder |
| `ulysses8-two-host.toml` | Four on each of two hosts | Ulysses8 denoiser, TP8 text encoder |
| `fasth3-ulysses4x2.toml` | Eight on one host | Two Ulysses4 replicas |
| `fasth3-ulysses4x2-two-host.toml` | Four on each of two hosts | One Ulysses4 replica per host |
| `fasth3-dp8-text-tp8.toml` | Eight on one host | Eight one-GPU replicas, shared TP8 text encoder |
| `gather8.toml` | Eight on one host | All-gather sequence-parallel denoiser, TP8 text encoder |

One replica spanning every GPU gives the lowest latency; replicas serve more requests at once. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#deployments) lists the recommended deployment and options for each GPU type and goal, and describes two-host startup.

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process and route readiness |
| `GET /metrics` | Runtime metrics |
| `GET /version` | Build information |
| `GET /v1/models` | Configured served model |
| `GET /v1/capabilities` | Served tasks, canvases, duration limits, schedule, accepted request fields and job retention |
| `POST /v1/videos/sync` | Generate one video and return the MP4 |
| `POST /v1/videos` | Create an asynchronous video job |
| `GET /v1/videos`, `GET /v1/videos/{id}` | List jobs, or read one job's state and progress |
| `GET /v1/videos/{id}/content` | Download a completed job's MP4 |
| `DELETE /v1/videos/{id}` | Cancel or delete a job |
| `POST /v1/chat/completions` | Streaming and non-streaming text, image-input, image-output, and interleaved generation |
| `POST /v1/images/generations` | Single-image generation adapter for configured omni descriptions |
| `POST /v1/systemone` | TypeSafe System One decision readout (DiffusionGemma) |

The server registers the routes of the tasks the served model performs: the video routes and `/v1/capabilities` for a video model, chat completions for every other model, image generation for an omni model, and `/v1/systemone` for DiffusionGemma. Any other path answers `404` with `{"detail": "Not Found"}`. With an API key configured, every route but `/health` and `/metrics` requires it. On SIGTERM or Ctrl-C the server stops accepting connections and lets requests in flight, streamed responses included, finish within `server_config.shutdown_timeout`; asynchronous video jobs end at once, since a restarted server keeps none.

DiffusionGemma readout settings are fixed when the server starts. `readout_config.canvas = "full"` uses the checkpoint's full canvas; `compact` rounds each answer scaffold up to a multiple of 16. A numeric value, such as `--set readout_config.canvas=64`, fixes every canvas to that length and splits larger question sets across complete canvases. Numeric lengths must be positive multiples of 16 no greater than the checkpoint's canvas length. `readout_config.candidates = "variants"` sums the supported token spellings of each answer; `primary` reads only the space-prefixed spelling (` A`, ` B`, ` yes`, ` no`, and so on). Defaults are `full` and `variants`. Canvas length and candidate selection change the returned distribution and must match between systems in a numerical or performance comparison.

See the [DiffusionGemma serving guide](docs/diffusion_gemma/serving.md) for checkpoint setup, four-GPU serving, decision and chat examples, precision choices, and measurement contracts.

A video request is the MiniMax-H3 request body: `model`, `prompt`, `task` (`t2va`), `target` with `short_edge` 768, `aspect_ratio` `16:9` and `duration_seconds` (4 to 15), and an optional `seed` (default 42). The [FastH3 guide](docs/minimax_h3/minimax_h3.md#generate-a-video) lists every field. Model discovery returns exactly one entry with the standard `id`, `object`, `created`, and `owned_by` fields; `id` is the configured served-model name.

The metrics endpoint publishes scheduler, worker, request and HTTP metrics in one OpenMetrics response. Each finished request is counted in `uniserve:request_success_total` under its `finished_reason` (`stop`, `length`, `abort`, `error`), with its latency, queue time and token counts in the request histograms, labeled by model name and the data-parallel rank (`engine`) that served it; running and waiting requests are the scheduler gauges `uniserve:num_requests_running` and `uniserve:num_requests_waiting`.

## Serving options

UniServe's options form one tree, following TensorRT-LLM's serving configuration: the options most deployments state sit at its top level, and the rest in `*_config` tables, one per part of the system. `uniserve serve` reads the tree from a `--config` TOML file, from flags for the most common options, and from repeatable `--set PATH=VALUE` assignments that reach any option by its path, each overriding the last:

```bash
uniserve serve "$H3_MODEL" --worker-groups configs/minimax_h3/ulysses4.toml \
  --served-model-name FastH3 --max-running-requests 2 \
  --set cuda_graph_config.policy=off \
  --set quantization_config.mode=balanced
```

| Flags | Options |
| --- | --- |
| `MODEL`, `--served-model-name`, `--max-model-len` | The checkpoint, its served name and context length |
| `--host`, `--port` | The HTTP listener (`server_config`) |
| `--worker-groups`, `--tensor-parallel-size`, `--data-parallel-size`, `--host-identity` | Placement: a deployment file, or the tensor- and data-parallel shorthand |
| `--max-running-requests`, `--mem-fraction-static` | Resident requests and each GPU worker's device-memory share |
| `--worker-executable` | The `uniserve` executable the workers run as (`worker_config.executable`) |
| `--config`, `--set`, `--print-config` | The configuration file, assignments by path, and printing the resolved tree |

| Table | Options for |
| --- | --- |
| `server_config` | HTTP listener, API key, timeouts and HTTP admission |
| `load_config`, `kv_cache_config`, `scheduler_config` | Checkpoint loading, the KV cache, and admission and batching |
| `placement_config`, `moe_config` | Placement without a deployment file and product transport; expert parallelism |
| `cuda_graph_config`, `flashinfer_config`, `worker_config` | CUDA graph capture, the FlashInfer provider, and the worker processes |
| `logging_config` | Request and statistics logs |
| `chat_config`, `image_fetch_config` | Chat models: chat template, reasoning parser and image URL fetching |
| `video_config`, `media_config`, `video_job_config` | MiniMax-H3: canvases, durations, prompt capacities, condition media and asynchronous jobs |
| `readout_config`, `diffusion_generation_config` | DiffusionGemma readouts and block-diffusion sampling |

A model-family table is refused at startup when the served checkpoint belongs to another family. `--print-config` prints the resolved tree, defaults included, in the form `--config` reads; `--log-level` and `--log-level-http` set the process's log levels. [Configuration](docs/configuration.md) describes every option, the configuration file and the deployment-file format.

On a GPU, startup captures CUDA graphs for decode steps and for prefill steps before the server reports ready, and serving replays them. Prefill graphs hold up to `max_num_batched_tokens` prompt tokens (plus one image's feature tokens for models that read images) and up to 31 prompts; decode graphs hold up to 128 requests. Both hold at most `max_running_requests` and at most as many requests as the KV pool holds a page of every cache group for. A prefill or decode step that no captured graph holds runs eagerly. Larger token budgets capture more graphs, so startup takes longer. Startup also runs every image encoder and decoder once, so the `uniserve-kernel-table` line it logs names the kernel of every call the server makes. `cuda_graph_config.policy = "off"` serves every call without graphs, and so does `auto` with `attention_backend = "torch"`, whose captured calls attend over every row's whole capacity.

Run `uniserve serve --help` for the flags and `uniserve serve --print-config` for every option.

## Reproduce the measurements

The serving evaluator runs HTTP workloads against a server, validates every response, and writes reproducible result bundles. `uniserve_eval/fast_h3.toml` holds the deployments and workload of the UniServe FastH3 post, and `uniserve_eval/fast_h3_h200.toml` the H200 measurements in the FastH3 guide. The `bench` extra installs the evaluator; `uv sync` keeps exactly the extras it names, so name `gpu` beside it:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra gpu --extra bench
export UNISERVE_FAST_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run gb200-4-bf16
```

The [evaluation guide](docs/minimax_h3/evaluation.md) lists every suite and the two-host procedure.

## Text and expert parallelism

For data-parallel serving, run one scheduler per GPU; a model that tensor parallelism cannot split, such as DiffusionGemma, serves four GPUs this way:

```bash
uniserve serve /models/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma \
  --data-parallel-size 4
```

Adding `--set moe_config.expert_parallel=true` keeps the four schedulers' attention data-parallel and shards each expert layer across them, so every GPU holds a quarter of the experts and the schedulers exchange tokens at each expert layer. Every expert layer then runs in steps the schedulers take together: a scheduler without work joins each step another one starts, and all of them pad a step to the largest one's captured graph. With an NVFP4 checkpoint, `moe_config.exchange = "megamoe"` fuses each expert layer's exchange and expert computation into one kernel.

For text-to-video-and-audio generation with the FastH3 checkpoints, including the packed NVFP4 releases, use the [FastH3 guide](docs/minimax_h3/minimax_h3.md).

## Development and verification

The `justfile` exposes the canonical repository checks:

```bash
just lint
just test-rust
just test-python-fast
just test-python-integration
just test-python-cuda
just test-python-e2e
```

The fast and integration suites run without a CUDA device; tests that need one carry the `gpu` marker, which `test-python-cuda` selects at the unit and integration layers. The FastH3 GPU tests load a complete checkpoint on four GPUs:

```bash
UNISERVE_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2 \
  .venv/bin/python -m pytest \
  tests/python/integration/model_loading/test_h3_parallel_latents.py \
  tests/python/e2e/test_h3_parallel_http.py
```

The serving evaluator runs HTTP workloads against serving models, measures performance, validates response correctness, and writes reproducible result bundles. Benchmark points are defined in [`uniserve_eval/profiles.toml`](uniserve_eval/profiles.toml).

For decision-session replay, use the `systemone` dataset with a non-empty `session_id` on every JSONL row. The evaluator preserves file order within each session and sends the next decision only after that session's previous response. Set `load.request_rate = inf`; the sessions run concurrently, one request in flight per session. A session trace has no repeated-row warmup: each initial cold request is included in the measured trace. Session identifiers are benchmark metadata and are not sent to the model. Rows without session identifiers retain the seeded shuffled workload behavior.

## Repository layout

The [architecture guide](docs/architecture.md) describes how the daemons, the head and the rank processes divide the work, how a rank divides it between Rust and Python, and how a request travels through them.

```text
crates/models/                Model-family request semantics: the MiniMax-H3 video rules
crates/storage/               The Storage Server's protocol and client
crates/storage-server/        The Storage Server and its Mooncake connector
crates/observability/         Runtime metrics and process registry
crates/ipc/                   Engine-to-rank messages, serialization, rank channels and the launch descriptor
crates/cuda/                  CUDA driver, VMM, NVML and NCCL bindings loaded at runtime
crates/engine/                Scheduler, KV allocation, executors, worker fleet and rank launch
crates/server/                Serving options, model profiles, serving funnel, HTTP API, and engine clients
crates/worker/                Rank runtime: the library a rank process loads through TVM-FFI
  launch/, config/            Launch descriptor, rank configuration and startup
  model/                      Model description, loading and component placement
  executor/, runner/          Batch execution per computation and its CUDA graphs
  request/, sampling/         Request progress and sampling state
  memory/                     KV, latent, tensor, request-slot and output backing
  transfer/                   Local, shared-memory, CUDA VMM and channel transfers
  parallel/                   Host groups, topology and collectives across ranks
  stream/, host/, capacity/   CUDA streams and events, host lanes, capacity planning
crates/uniserve/              The `uniserve` executable: `serve`, `worker` and `storage`
uniserve/                     Numerical layers, loading, resource binding, concrete models and checkpoint catalog
  runner/                     The Python half of each runner, which the worker runtime calls
  worker.py                   Rank process entry: loads the worker runtime and hands it the library's import function
uniserve_kernels/             Native CUDA, Triton and CuTe kernels
uniserve_eval/                Serving evaluator, profiles, and request workloads
configs/                      Deployment files: H3 (`minimax_h3/`) and dedicated experts (`afd/`)
docs/architecture.md          Processes, the rank planes and the request path
docs/configuration.md         The serving option tree, configuration files and deployment files
docs/minimax_h3/              H3 serving, library and evaluation guides, container files
```

## License

Apache-2.0.
