# UniServe

UniServe serves FastH3 8-Step text-to-video-with-audio and keyframe-conditioned generation on NVIDIA Blackwell and Hopper GPUs. Each request returns a finished MP4: 1344×768 H.264 video at 24 fps with stereo 32-kHz AAC audio. The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) describes the design and its measurements, and the [FastH3 and MiniMax-H3 guide](docs/minimax_h3/minimax_h3.md) covers every deployment, precision and option. The same guide covers the base MiniMax-H3 checkpoint's text-to-video, keyframe and reference requests and FastH3 OmniRef reference requests.

UniServe is a Rust serving system. Rust owns the control and data planes and the numerical computation alike: HTTP admission, scheduling, the main processes, request state, storage, CUDA streams and graph policy, transfers and response assembly. No serving process carries a Python interpreter: Python appears in the Build Server alone, as the compiler of Triton and CuTe DSL kernel packages, and in the evaluation client `uniserve_eval`.

The executable serves two model families, Qwen3-MoE (chat) and MiniMax-H3 (video generation), each registered once in `src/main.rs`. A server resolves its checkpoint to the family that claims the class it declares, serves exactly that model and exposes one served-model identity.

## Requirements

| Item | Requirement |
| --- | --- |
| OS / architecture | Linux on x86-64 or aarch64 |
| GPU | NVIDIA Hopper or Blackwell GPUs; H200, GB200 and RTX PRO 6000 Blackwell Server Edition have end-to-end validation |
| CUDA | CUDA 13 toolkit and a driver of CUDA 13.0 or later (`libcuda.so.1`, `libnvidia-ml.so.1`) |
| NCCL | `libnccl.so.2` of release 2.30.4 or later on the default loader path |
| CUPTI | `libcupti.so.13` of release 13.4.92 or later on the default loader path |
| FFmpeg | FFmpeg 8.1 with libx264, as shared libraries that pkg-config and the loader find |
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

The sync builds the `uniserve` executable. The native libraries come from the host, where the standard mechanisms find them, and nothing points the build or the processes at them:

- The NVIDIA driver, NCCL and CUPTI are opened at run time by their sonames from the default loader path, as NVIDIA's distribution packages install them (`/etc/ld.so.conf.d` registers any other directory that holds them). A process refuses a library older than the release the table above names with an error that names it, before it uses it.
- FFmpeg 8.1 is found at build time through pkg-config, at exactly the library versions of that release: a distribution package, or an installation prefix whose `lib/pkgconfig` the build's `PKG_CONFIG_PATH` names and whose `lib` the loader configuration (`/etc/ld.so.conf.d`, then `ldconfig`) registers. A host whose distribution lacks FFmpeg 8.1 builds the FFmpeg 8.1.2 release and libx264 into such a prefix with their own build systems, configured as [the media crate](crates/media/README.md) configures them.
- A build that runs from its target directory, such as a development or CI build, can instead build the pinned FFmpeg 8.1.2 and libx264 from source with `--features media/vendored` (it needs curl, tar, xz, make, a C compiler and, on x86-64, nasm). Its executables find those libraries through their run path into the target directory.

The Build Server keeps an environment of its own, which `uv sync --locked --python /usr/bin/python3.12 --project crates/build-server/python` creates; it compiles the kernel packages with that environment and the host's CUDA toolkit. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#install) lists the system packages and a container build.

For RDMA transfers between hosts, install the pinned UCX and NIXL builds described in [NIXL transfers](docs/nixl.md).

## Quickstart

Download the checkpoint and serve it on four GPUs:

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir "$H3_MODEL"

uniserve serve --model-dir "$H3_MODEL" \
  --config configs/minimax_h3/ulysses4.toml \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2 \
  --max-model-len 2048
```

A video deployment states `--max-model-len`, the longest prompt it admits in tokens: the pipeline keeps no key/value cache from which a length would follow. Startup captures prompt capacities up to it (`video_config.text_capacities`, by default 1024 and then steps of 2048), the length itself the largest.

Startup prepares and captures every admitted request shape before `/health` reports ready; on four GB200 GPUs this takes about 11 minutes. The first startup on a machine also compiles and caches the kernels FastH3 runs and takes about 19 minutes. Then generate a video:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","size":"1344x768","seconds":5,"seed":1000}' \
  --output forest.mp4
```

## Deployments

A deployment file, a configuration file passed to `--config`, places FastH3's blocks on devices and hosts: TOML `[[worker_groups]]` groups, each with its ranks (`cuda:0-3`), replicas and blocks. `configs/minimax_h3/` holds them; the FastH3 deployments with published measurements are:

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
| `GET /v1/capabilities` | The served model's video rules (tasks, sizes, durations, frame rate, schedule), prompt capacity, media policy, accepted request fields and job retention |
| `POST /v1/videos/sync` | Generate one video and return the MP4 |
| `POST /v1/videos` | Create an asynchronous video job |
| `GET /v1/videos`, `GET /v1/videos/{id}` | List jobs, or read one job's state and progress |
| `GET /v1/videos/{id}/content` | Download a completed job's MP4 |
| `DELETE /v1/videos/{id}` | Cancel or delete a job |
| `POST /v1/chat/completions` | Streaming and non-streaming chat generation, with reasoning and tool-call parsing where the model's chat profile names them |
| `/admin/*` | The admin API, served apart from the routes above on the daemon's operator and harness Unix sockets: configuration changes (`PUT /admin/config`), each applied as the least it needs and, when it moves modules, as an update on trial that fails no request, their commit and rollback, and the other operations of a serving deployment ([docs/admin.md](docs/admin.md)) |

The server registers the routes of the tasks the served model performs: the video routes and `/v1/capabilities` for a video model, and chat completions for a text model. Any other path answers `404` with `{"detail": "Not Found"}`. With an API key configured, every route but `/health` and `/metrics` requires it. Another build of the engine and workers is a configuration change of `worker_config.executable` (`PUT /admin/config` with `{"set": ["worker_config.executable=/path/to/uniserve"]}`): it answers once the new epoch serves on trial, with `pause_seconds`, how long requests waited for it, and `POST /admin/config/commit` ends the trial. On SIGTERM or Ctrl-C the server stops accepting connections and lets requests in flight, streamed responses included, finish within `server_config.shutdown_timeout`; asynchronous video jobs end at once, since a restarted server keeps none.



A video request has the same fields for every video model: `model`, `prompt`, `negative_prompt`, `size` (`WxH`), `seconds` (a number) or `num_frames`, `fps`, `seed`, `num_inference_steps`, `guidance_scale`, `input_references` (image, video or audio media, each optionally anchoring a generated frame with `frame_index`) and `extra_params`; any other field is refused. The served model's processor holds a request to its own rules: MiniMax-H3 derives its task from the references, serves the sizes the deployment prepares (1344x768 and 768x1344 by default), aligns `seconds` (4 to 15) to its frame rule at 24 fps and uses seed 42 when none is given. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#generate-a-video) lists every field and MiniMax-H3's mapping. Model discovery returns exactly one entry with the standard `id`, `object`, `created`, and `owned_by` fields; `id` is the configured served-model name.

A chat request takes OpenAI's chat-completions fields and refuses any other: `messages` (system, developer, user, assistant and tool roles, with `image_url` parts for a model that takes images, whose `detail` is `auto` or absent because the model's image processor sizes every image), `tools` (function definitions, with `strict`), `tool_choice` (`auto`, `none`; `required` and a named function need constrained decoding and are refused by name), `max_completion_tokens` or OpenAI's deprecated name of it `max_tokens` (a request states one of the two), `stop`, `stream` with `stream_options.include_usage`, `logprobs` and `top_logprobs` (at most the deployment's `max_logprobs`), the sampling controls, `reasoning_effort`, `chat_template_kwargs`, `priority` and `cache_salt`, which keeps a conversation's requests in one prefix-cache partition. A response states parsed reasoning as `reasoning` (in a stream, `delta.reasoning`), and an assistant message sent back in the history states it under the same name. A model's tool calls are parsed in the format its family's chat profile declares, JSON inside `<tool_call>` tags (Qwen 2.5 and Qwen3) or function and parameter tags (Qwen3.6 and Qwen3.8 Flash Next), and returned as `tool_calls`: in a stream, each call's `index`, `id`, `type` and function `name` once with empty `arguments`, then its `arguments` in increments, and `finish_reason: "tool_calls"`. A tag call's values are typed by the JSON Schema of its tool's parameter: the text when the schema accepts a string, else the JSON the text states. A call that breaks its format's grammar, states a value its schema refuses, or is cut off by the end of the output fails the response as a server error instead of reaching the client as text. A family whose profile declares no tool-call format refuses a request with tools. A response's `usage` (in a stream, the usage chunk `stream_options.include_usage` asks for) counts `prompt_tokens`, `completion_tokens` and `total_tokens`, and as `prompt_tokens_details.cached_tokens` the prompt tokens the request's first admission found cached in any tier (device, host, peer, remote store) and did not compute: fewer than its prompt's, whose last token is always computed.

The metrics endpoint publishes scheduler, worker, request and HTTP metrics in one OpenMetrics response. Each finished request is counted in `uniserve:request_success_total` under its `finished_reason` (`stop`, `length`, `abort`, `error`), with its latency, queue time and token counts in the request histograms, labeled by model name and the data-parallel rank (`engine`) that served it; running and waiting requests are the scheduler gauges `uniserve:num_requests_running` and `uniserve:num_requests_waiting`; and a deployment of prefill and decode groups counts the requests moved to a decode group in `uniserve:request_transfers_total`, the moves that failed in `uniserve:request_transfers_failed_total`, and the blocks the decode groups received in `uniserve:request_transfer_blocks_total`; what a shortage of room took is counted as the running requests preempted (each ended as migratable and resubmitted) in `uniserve:scheduler_preempted_requests_total`, the cached blocks evicted from the device and host regions in `uniserve:scheduler_evicted_blocks_total`, and the draft-cache blocks running requests gave up in `uniserve:scheduler_given_up_draft_blocks_total`.

## Serving options

UniServe's options form one tree, following TensorRT-LLM's serving configuration: the options most deployments state sit at its top level, and the rest in `*_config` tables, one per part of the system. `uniserve serve` reads the tree from a `--config` TOML file, from flags for the most common options, and from repeatable `--set PATH=VALUE` assignments that reach any option by its path, each overriding the last:

```bash
uniserve serve --model-dir "$H3_MODEL" --config configs/minimax_h3/ulysses4.toml \
  --served-model-name FastH3 --max-running-requests 2 \
  --set 'quantization.mode="balanced"'
```

| Flags | Options |
| --- | --- |
| `MODEL`, `--served-model-name`, `--max-model-len` | The checkpoint, its served name and context length |
| `--host`, `--port` | The HTTP listener (`server_config`) |
| `--tensor-parallel-size`, `--data-parallel-size`, `--host-identity` | Parallel configuration without a deployment file: the tensor- and data-parallel shorthand |
| `--max-running-requests`, `--mem-fraction-static` | Resident requests and each GPU worker's device-storage share |
| `--config`, `--set` | The configuration files (a deployment file among them) and assignments by path |

| Table | Options for |
| --- | --- |
| `server_config` | HTTP listener, API key, timeouts and HTTP admission |
| `load_config`, `kv_cache_config`, `scheduler_config` | Checkpoint loading, the KV cache, and admission and batching |
| `device_config`, `moe_config` | Parallel configuration without a deployment file, and product transport; expert parallelism |
| `cuda_graph_config`, `worker_config` | CUDA graph capture, and the main processes and their workers |
| `logging_config` | Request and statistics logs |
| `chat_config`, `image_fetch_config` | Chat models: chat template, reasoning parser and image URL fetching |
| `video_config`, `media_config`, `video_job_config` | Video models: output sizes, durations, prompt capacities, condition media and asynchronous jobs |

A chat option or table is refused at startup when the served checkpoint is a video model, and a video table when it is a chat model. `GET /admin/config` returns the resolved tree in the form `--config` reads; `--log-level` and `--log-level-http` set the process's log levels. [Configuration](docs/configuration.md) describes every option, the configuration file and the deployment-file format.

On a GPU, startup captures CUDA graphs for decode steps and for prefill steps before the server reports ready, and serving replays them. Prefill graphs hold up to `max_num_batched_tokens` prompt tokens and up to 31 prompts; decode graphs hold up to 128 requests. Both hold at most `max_running_requests` and at most as many requests as the KV pool holds a page of every cache group for. A prefill or decode step that no captured graph holds runs eagerly. Larger token budgets capture more graphs, so startup takes longer. Startup also runs every image encoder and decoder once, so the `uniserve-kernel-table` line it logs names the kernel of every call the server makes. With `attention_backend = "torch"` every call serves without graphs, since its captured calls would attend over every row's whole capacity.

Run `uniserve serve --help` for the flags; [Configuration](docs/configuration.md) lists every option.

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

For data-parallel serving, run one scheduler per GPU.

Adding `--set moe_config.expert_parallel=true` keeps the four schedulers' attention data-parallel and shards each expert layer across them, so every GPU holds a quarter of the experts and the schedulers exchange tokens at each expert layer. Every expert layer then runs in steps the schedulers take together: a scheduler without work joins each step another one starts, and all of them pad a step to the largest one's captured graph. With an NVFP4 checkpoint, `moe_config.exchange = "megamoe"` fuses each expert layer's exchange and expert computation into one kernel.

For text-to-video-and-audio generation with the FastH3 checkpoints, including the packed NVFP4 releases, use the [FastH3 guide](docs/minimax_h3/minimax_h3.md).

## Development and verification

The `justfile` exposes the canonical repository checks:

```bash
just lint
just test-rust
just test-python-fast
just test-python-gpu
```

`test-python-fast` runs without a CUDA device. The HTTP tests under `tests/python/e2e` start real servers on the host's GPUs; `test-python-gpu` builds the release executable once and runs them one case per free GPU, and a plain `pytest` skips them unless `UNISERVE_RUN_GPU_E2E=1`. The host's checkpoints come from the environment, and a test whose checkpoint is unset is skipped, naming the variable. [Testing](docs/testing.md) lists the variables; the FastH3 tests, for example, load a complete checkpoint on four GPUs:

```bash
UNISERVE_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2 \
  tools/gpu_e2e.sh -m "e2e and gpu" tests/python/e2e/test_h3_parallel_http.py
```

The serving evaluator runs HTTP workloads against serving models, measures performance, validates response correctness, and writes reproducible result bundles. Benchmark points are defined in [`uniserve_eval/profiles.toml`](uniserve_eval/profiles.toml).

For decision-session replay, give every row of a JSON Lines workload a non-empty `session_id`. The evaluator preserves file order within each session and sends the next decision only after that session's previous response. Set `load.request_rate = inf`; the sessions run concurrently, one request in flight per session. A session trace has no repeated-row warmup: each initial cold request is included in the measured trace. Session identifiers are benchmark metadata and are not sent to the model. Rows without session identifiers retain the seeded shuffled workload behavior.

A point whose `load.duration_s` is set runs a timeline load: it issues requests from the load's start for that many seconds, request `n` taking row `n mod num_prompts`, as a closed loop of `max_concurrency` streams when `request_rate = inf`, and then waits for every stream to end. Its `[[benchmarks.<name>.changes]]` act on the deployment while the load runs, each `at_s` seconds after the start or `delay_s` after an earlier change named by `after` (counted, with `phase`, from the admin event that reports the update entering that phase): an HTTP call (`method`, `path`, `body`) to the admin listener the server states as `admin_socket`, a `signal` to the one process of the deployment whose command line matches `process`, or a `command` run in the deployment's environment. Issuing continues while a change runs and for `load.settle_s` after the last one ended. Each record of `requests.jsonl` holds its chunks' arrival times and, when the request asks for them (`sampling.extra_body` with `logprobs`, `top_logprobs`, `return_token_ids` and `return_tokens_as_token_ids`), its token ids and their candidates; `timeline.json` holds the load's clock, each change's answer with its times and the admin events. `uniserve-eval check-change <run> --control <control> --check-metric <counter> --replay <verdict.json>` checks such a run against the same load without changes: no request failed or lost its connection, each stream carried exactly its requested tokens and agrees with the control's output of its row by the near-tie rule of vLLM's `check_logprobs_close`, each request submitted during the change received its first token by the change's end plus the control's 99th-percentile time to first token, the named check-failure counter reads 0 and the replay verdict is `pass`. It writes `result.md` with these items and the longest inter-token gap and time to first token beside the control's, and prints them as one JSON line.

A change may also be a `signal` with a `host`, sent to the one process of this user on that host whose command line matches `process`, found and signalled there over `ssh` in batch mode; or a `cancel` of that many streams of a closed loop at once, the earliest issued or with `cancel_order = "newest"` the latest issued, which the client disconnects mid-stream: their requests are recorded with the classifier `cancelled`, the streams go on, and the point leaves them out of its metrics and validation (`cancelled_count`) and `check-change` out of its items. A change that moves the deployment to another configuration names it (`configuration`, a rollback naming the one it returns to), and each `--control-of NAME=PATH` gives `check-change` the control of a configuration, started in it: a request agrees with the control of the configuration it was submitted on or of one a change overlapping its life moved to, since a request in flight across an epoch's switch may end on either side. `--switch NAME --block-size N` adds a check of the cache a switch carries over: the first request submitted after the switch named `NAME` ended, and before the next change, of each prompt that received a token before the switch, reports as `usage.prompt_tokens_details.cached_tokens` at least the whole blocks of its prompt less the last token, which an admission always computes. `uniserve-eval check-faults <run> --reference <alone> --check-metric <counter> --replay <verdict.json>` checks a run whose changes are faults against its rows served alone, the bundle of a point with `max_concurrency = 1` and no `duration_s`, which sends each row once with nothing else running: every request not cancelled answered 200 with its requested tokens, which agree with its row's by the near-tie rule, and the invariants' evidence holds. A timeline load also scrapes every replica's `/metrics` once a second while it runs, into `scrapes.jsonl` with each scrape's send and arrival times. `uniserve-eval check-stop <run> --control <control> --headroom-tokens N` checks a run in which a process of the deployment was stopped with `SIGSTOP` and continued with `SIGCONT` against the same load without the signals: no request failed; in each stop window the gaps between tokens that ended in it have their p50 and p99 within the control's gaps of the same window, and no part of it without a token outlasts the control's longest gap; the counter of calls left out for late blocks rises only where a stop window explains it, after some stream decoded `N` tokens in it, and every request in flight at a window's end completes.

## Repository layout

The [architecture guide](docs/architecture.md) describes how the daemons, the engine process and its workers divide the work, how a main process's threads divide it, and how a request travels through them.

```text
src/                    The root package, `uniserve`: the command line, logging, the `serve` and `keys` commands, and `main.rs`, which dispatches every command; the base executable holds the daemon, the Storage Server and the Build Server, and links no engine or model
crates/engine/          The main process of a node and epoch: its control thread, one scheduler per data-parallel rank, the executor each scheduler hands its batches to, the batch vocabulary, the peer link, one worker thread per device with its runners, graphs and memory owners, the client of the Storage Server, and the catalogue that resolves a checkpoint to its model family
crates/proto/           The interfaces across builds: the request interface and the process interface (Protobuf), and the video condition vocabulary
crates/scheduling/      The scheduling interface between the scheduler's mechanism and its policy, and the policy package under `policies/` (`arrival`), compiled to WebAssembly
crates/storage/         The Storage Server's protocol and client
crates/storage-server/  The Storage Server, its Mooncake connector and its Transfer Server (with NIXL, loaded at runtime)
crates/platform/        The CUDA boundary: driver, virtual memory, device memory and allocator, streams, events, graphs, kernels and NCCL
crates/tensor/          The tensor library: `Tensor`, element types, shapes and layouts, devices and storage, the CPU backend and the interface every other device backend implements; it knows nothing of UniServe
crates/ops/             The device side of the tensor library and the interface between Rust and kernel packages: operators, their descriptors and descriptor keys, launch code and every kernel, one package per operator and backend under `kernels/`, and the CUDA and meta backends of `tensor` with the allocator of their memory
crates/nn/              The neural-network library: modules and fused operators (each with its CPU formula), weight loading and quantized formats, diffusion schedules and solver formulas, tiled VAE computation, profiler ranges, the capability traits execution calls on a model and the family traits it loads a model through
crates/processing/      The request side of a model family: its chat profile (the reasoning and tool-call formats its checkpoints write) or its video processor, which checks and processes model-agnostic video requests, and the family-neutral frame, canvas, cover-crop and resampling rules; a family's own request rules are a module each (`qwen`, `qwen_vl`, `minimax_h3`)
crates/models/          The model families, one crate with a module per family, on `nn` and the tensor library
  src/qwen3/            The Qwen3 family: configurations, the decoder layer, `Qwen3MoeForCausalLM` and the Qwen3.5-MoE language model with its drafter, and the Qwen3-VL encoder and vision tower
  src/qwen4_exp/        Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`), on the vision tower and configuration readers of `qwen3`
  src/minimax_h3/       MiniMax-H3 and FastH3: checkpoint configuration, the diffusion transformer and its denoiser, the video and audio VAEs and the packed sequence; their request processing is `crates/processing/src/minimax_h3/`
crates/daemon/          The node's daemon: process supervision and the Storage Server claim; on the leader, serving options, the HTTP server and its connections, metrics, the served model, the engine's client, and the API server: the OpenAI-compatible API, input processing (templates, tokenization, media) and output processing (detokenization, parsers, response bodies)
crates/service/         Host services: the per-user socket, its owner lock, a detached start and the start of an absent service; the one connection every process interface runs on
crates/media/           H.264/AAC output encoding and MP4 muxing, conditional media reading and host resamplers
crates/build-server/  The Build Server: builds packages into modules, keeps them and hands them out; one host service per host and user
crates/testkit/         Test doubles at the engine's seams: the executor of a worker group (`FakeExecutor`), the Storage Server's buffer pool in process (`InProcessPool`) and a fake engine over both; never in a serving build
crates/manager/         `uniserve-manager`: an optional multi-node manager, a client of each node's `/admin`
uniserve_eval/          Serving evaluator, profiles, and request workloads
configs/                Deployment files: H3 (`minimax_h3/`) and dedicated experts (`afd/`)
docs/architecture.md    Processes, the rank planes and the request path
docs/configuration.md   The serving option tree, configuration files and deployment files
docs/admin.md           The admin API: a node's status, event log, deployment and configuration
docs/profiling.md       Timeline and native profiling
docs/minimax_h3/        H3 serving and evaluation guides, container files
docs/qwen3_moe/         Qwen3-MoE serving and its GSM8K quality confirmation
```

## License

Apache-2.0.
