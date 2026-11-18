# UniServe

UniServe serves FastH3 8-Step text-to-video-with-audio and keyframe-conditioned generation on NVIDIA Blackwell and Hopper GPUs. Each request returns a finished MP4: 1344×768 H.264 video at 24 fps with stereo 32-kHz AAC audio. The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) describes the design and its measurements, and the [FastH3 and MiniMax-H3 guide](docs/minimax_h3/minimax_h3.md) covers every deployment, precision and option. The same guide covers the base MiniMax-H3 checkpoint's text-to-video, keyframe and reference requests and FastH3 OmniRef reference requests.

UniServe is a Rust serving system. Rust owns the control and data planes and the numerical computation alike: HTTP admission, scheduling, the main processes, request state, storage, CUDA streams and graph policy, transfers and response assembly. No serving process carries a Python interpreter: Python appears in the Build Server alone, as the compiler of Triton and CuTe DSL kernel packages, and in the Python tests.

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
| libclang | libclang 18 or later with its headers, which bindgen loads at build time (`libclang-18-dev` on Ubuntu 24.04) |
| Python | Python 3.12, which uv installs when the host has none |
| uv | uv, which creates the locked Python environments |
| Rust | rustup, which installs the Rust release `rust-toolchain.toml` pins, with its components and the wasm32 target |

## Installation

Install uv and rustup from the host's packages or their releases, then run from the repository root:

```bash
rustup toolchain install
```

`rustup toolchain install` installs the release `rust-toolchain.toml` pins. `uv run uniserve ...` from the repository root builds the `uniserve` executable into the repository's environment (`.venv`) the first time, and again after a change, before it runs it; `uv sync --locked` builds it alone. uv selects the interpreter by the project's `requires-python`. The native libraries come from the host, where the standard mechanisms find them, and nothing points the build or the processes at them:

- The NVIDIA driver, NCCL and CUPTI are opened at run time by their sonames from the default loader path, as NVIDIA's distribution packages install them (`/etc/ld.so.conf.d` registers any other directory that holds them). A process refuses a library older than the release the table above names with an error that names it, before it uses it.
- FFmpeg 8.1 is found at build time through pkg-config, at exactly the library versions of that release: a distribution package, or an installation prefix whose `lib/pkgconfig` the build's `PKG_CONFIG_PATH` names and whose `lib` the loader configuration (`/etc/ld.so.conf.d`, then `ldconfig`) registers. A host whose distribution lacks FFmpeg 8.1 builds the FFmpeg 8.1.2 release and libx264 into such a prefix with their own build systems, configured as [the media crate](crates/media/README.md) configures them.
- A build that runs from its target directory, such as a development or CI build, can instead build the pinned FFmpeg 8.1.2 and libx264 from source with `--features media/vendored` (it needs curl, tar, xz, make, a C compiler and, on x86-64, nasm). Its executables find those libraries through their run path into the target directory.

A deployment names the source tree it serves from in its policy package (`build_config.source`). The host's Build Server, which the daemon starts, creates its own environment from that tree with uv, the locked compilers of the kernel packages ([its README](crates/build-server/python/README.md)), and installs the tree's engine, scheduling policy and kernel packages before it serves; it compiles the kernel packages with that environment and the CUDA toolkit NVIDIA's packages register with pkg-config. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#install) lists the system packages.

For RDMA transfers between hosts, install the pinned UCX and NIXL builds described in [NIXL transfers](docs/nixl.md).

## Quickstart

A deployment is a policy package: a directory whose `config.toml` states the checkpoint, the serving options and the layout of its blocks on devices, and whose name names the deployment. Build and serve the checkpoint on four GPUs, from the repository root, with the layout of the package `configs/minimax_h3/ulysses4`:

```bash
mkdir fasth3 && cat > fasth3/config.toml <<'TOML'
model = "FastVideo/FastVideo-FastH3-8-Step-V2"
served_model_name = "FastH3"
max_running_requests = 2
max_model_len = 2048
admin_config = { host = "0.0.0.0", port = 8000 }
build_config = { source = "." }
TOML
sed -n '/^\[\[worker_groups\]\]/,$p' configs/minimax_h3/ulysses4/config.toml >> fasth3/config.toml
uv run uniserve serve fasth3
```

The model is a Hugging Face repository, `<owner>/<name>[@<revision>][/<folder>]`: the server reads it from the standard Hugging Face cache and downloads what the cache lacks, at the repository's default branch unless a revision is named. A host without network access populates the cache beforehand and sets `HF_HUB_OFFLINE`; `model_dir` in place of `model` serves a local checkpoint directory ([Configuration](docs/configuration.md)).

A video deployment states `max_model_len`, the longest prompt it admits in tokens: the pipeline keeps no key/value cache from which a length would follow. Startup captures prompt capacities up to it (`video_config.text_capacities`, by default 1024 and then steps of 2048), the length itself the largest.

Startup prepares and captures every admitted request shape before `/health` reports ready; on four GB200 GPUs this takes about 11 minutes. The first startup on a machine also compiles and caches the kernels FastH3 runs and takes about 19 minutes. Then generate a video:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","size":"1344x768","seconds":5,"seed":1000}' \
  --output forest.mp4
```

## Deployments

A package's `[[worker_groups]]` tables place FastH3's blocks on devices and hosts, each group with its ranks (`cuda:0-3`), replicas and blocks. `configs/minimax_h3/` holds packages of FastH3 with each layout; those with published measurements are:

| Package | GPUs | Layout |
| --- | --- | --- |
| `ulysses4` | Four on one host | Ulysses4 denoiser, TP4 text encoder |
| `ulysses8` | Eight on one host | Ulysses8 denoiser, TP8 text encoder |
| `ulysses8-two-host` | Four on each of two hosts | Ulysses8 denoiser, TP8 text encoder |
| `fasth3-ulysses4x2` | Eight on one host | Two Ulysses4 replicas |
| `fasth3-ulysses4x2-two-host` | Four on each of two hosts | One Ulysses4 replica per host |
| `fasth3-dp8-text-tp8` | Eight on one host | Eight one-GPU replicas, shared TP8 text encoder |
| `gather8` | Eight on one host | All-gather sequence-parallel denoiser, TP8 text encoder |

One replica spanning every GPU gives the lowest latency; replicas serve more requests at once. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#deployments) lists the recommended deployment and options for each GPU type and goal, and describes two-host startup.

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process and route readiness |
| `GET /version` | Build information |
| `GET /v1/models` | Configured served model |
| `GET /v1/capabilities` | The served model's video rules (tasks, sizes, durations, frame rate, schedule), prompt capacity, media policy, accepted request fields and job retention |
| `POST /v1/videos/sync` | Generate one video and return the MP4 |
| `POST /v1/videos` | Create an asynchronous video job |
| `GET /v1/videos`, `GET /v1/videos/{id}` | List jobs, or read one job's state and progress |
| `GET /v1/videos/{id}/content` | Download a completed job's MP4 |
| `DELETE /v1/videos/{id}` | Cancel or delete a job |
| `POST /v1/chat/completions` | Streaming and non-streaming chat generation, with reasoning and tool-call parsing where the model's chat profile names them |
| `/admin/*` | The admin API, served apart from the routes above on the daemon's operator and harness Unix sockets: the deployment's policy package (`GET` and `PUT /admin/policy`), each change applied as the least it needs and, when it moves modules, as an update on trial that fails no request and commits by itself, and the other operations of a serving deployment ([docs/admin.md](docs/admin.md)) |

The server registers the routes of the tasks the served model performs: the video routes and `/v1/capabilities` for a video model, and chat completions for a text model. Any other path answers `404` with `{"detail": "Not Found"}`. With an API key configured, every route but `/health` requires it. Another build of the engine and workers is a revision of the engine package installed at the Build Server (`uniserve build-server install crates/engine`), which a package stating `worker_config.engine_revision` selects (`PUT /admin/policy`): the new epoch serves on trial for `update_config.trial` and commits by itself, and the `PUT` answers with the revision's record once it converged. On SIGTERM or Ctrl-C the server stops accepting connections and lets requests in flight, streamed responses included, finish within `server_config.shutdown_timeout`; asynchronous video jobs end at once, since a restarted server keeps none.



A video request has the same fields for every video model: `model`, `prompt`, `negative_prompt`, `size` (`WxH`), `seconds` (a number) or `num_frames`, `fps`, `seed`, `num_inference_steps`, `guidance_scale`, `input_references` (image, video or audio media, each optionally anchoring a generated frame with `frame_index`) and `extra_params`; any other field is refused. The served model's processor holds a request to its own rules: MiniMax-H3 derives its task from the references, serves the sizes the deployment prepares (1344x768 and 768x1344 by default), aligns `seconds` (4 to 15) to its frame rule at 24 fps and uses seed 42 when none is given. The [FastH3 guide](docs/minimax_h3/minimax_h3.md#generate-a-video) lists every field and MiniMax-H3's mapping. Model discovery returns exactly one entry with the standard `id`, `object`, `created`, and `owned_by` fields; `id` is the configured served-model name.

A chat request takes OpenAI's chat-completions fields and refuses any other: `messages` (system, developer, user, assistant and tool roles, with `image_url` parts for a model that takes images, whose `detail` is `auto` or absent because the model's image processor sizes every image), `tools` (function definitions, with `strict`), `tool_choice` (`auto`, `none`; `required` and a named function need constrained decoding and are refused by name), `max_completion_tokens` or OpenAI's deprecated name of it `max_tokens` (a request states one of the two), `stop`, `stream` with `stream_options.include_usage`, `logprobs` and `top_logprobs` (at most the deployment's `max_logprobs`), the sampling controls, `reasoning_effort`, `chat_template_kwargs`, `priority` and `cache_salt`, which keeps a conversation's requests in one prefix-cache partition. A response states parsed reasoning as `reasoning` (in a stream, `delta.reasoning`), and an assistant message sent back in the history states it under the same name. A model's tool calls are parsed in the format its family's chat profile declares, JSON inside `<tool_call>` tags (Qwen 2.5 and Qwen3) or function and parameter tags (Qwen3.6 and Qwen3.8 Flash Next), and returned as `tool_calls`: in a stream, each call's `index`, `id`, `type` and function `name` once with empty `arguments`, then its `arguments` in increments, and `finish_reason: "tool_calls"`. A tag call's values are typed by the JSON Schema of its tool's parameter: the text when the schema accepts a string, else the JSON the text states. A call that breaks its format's grammar, states a value its schema refuses, or is cut off by the end of the output fails the response as a server error instead of reaching the client as text. A family whose profile declares no tool-call format refuses a request with tools. A response's `usage` (in a stream, the usage chunk `stream_options.include_usage` asks for) counts `prompt_tokens`, `completion_tokens` and `total_tokens`, and as `prompt_tokens_details.cached_tokens` the prompt tokens the request's first admission found cached in any tier (device, host, peer, remote store) and did not compute: fewer than its prompt's, whose last token is always computed.

What the deployment measures is two SQL tables of its observation store, `stats` and `events`, read with `GET` or `POST /admin/query?q=<SQL>` on the admin sockets ([docs/admin.md](docs/admin.md)); the metric names follow the Prometheus conventions, and every row carries the configuration revision it was measured under. Each ended request is one `request` event (its arrival, prompt, prefix-hit and output tokens, time to first token, time per output token and outcome) and one count of `uniserve_requests_total` under its `outcome` (`stop`, `length`, `repetition`, `cache_capacity`, `abort`, `error`), with its latency, queue time and token counts in the request histograms, labelled by the replica (`replica`) and the data-parallel rank (`scheduler`) that served it; running and waiting requests are the gauges `uniserve_scheduler_running_requests` and `uniserve_scheduler_waiting_requests`; HTTP requests are counted in `uniserve_http_requests_total` and timed in `uniserve_http_request_duration_seconds` by method and matched route; a deployment of prefill and decode groups counts the requests moved to a decode group in `uniserve_request_transfers_total`, the moves that failed in `uniserve_request_transfer_failures_total` and the blocks the decode groups received in `uniserve_request_transferred_blocks_total`; what a shortage of room took is counted as the running requests preempted (each ended as migratable and resubmitted) in `uniserve_scheduler_preempted_requests_total`, the cached blocks evicted in `uniserve_kv_cache_evicted_blocks_total`, and the draft-cache blocks running requests gave up in `uniserve_scheduler_given_up_draft_blocks_total`; each engine process's start, report and exit is a `process` event; and each replica's running revision is `uniserve_replica_revision`.

## Serving options

UniServe's options form one tree, following TensorRT-LLM's serving configuration: the options most deployments state sit at its top level, and the rest in `*_config` tables, one per part of the system. The tree is a policy package's `config.toml`, and the package is the deployment's whole configuration: `uniserve serve PACKAGE` takes no option of it, and every change while it serves is the next package, applied whole with `PUT /admin/policy` and kept as a numbered revision of the deployment's state:

```toml
model = "FastVideo/FastVideo-FastH3-8-Step-V2"
served_model_name = "FastH3"
max_running_requests = 2
quantization = { mode = "balanced" }
hosts = ["gb200-a", "gb200-b"]   # a deployment of two nodes, the leader first
```

| Options | For |
| --- | --- |
| `model`, `model_dir`, `served_model_name`, `max_model_len` | The checkpoint, its served name and context length |
| `hosts` | The deployment's nodes by host name, the leader first; every node serves the same package |
| `tensor_parallel_size`, `data_parallel_size` | Parallel configuration without a layout: the tensor- and data-parallel shorthand |
| `max_running_requests`, `mem_fraction_static` | Resident requests and each GPU worker's device-storage share |
| `worker_groups` | The layout: worker groups placed on devices and hosts, and the blocks each runs |

| Table | Options for |
| --- | --- |
| `server_config`, `admin_config` | API key, timeouts and HTTP admission; the HTTP listener and the port other nodes join at |
| `load_config`, `kv_cache_config`, `scheduler_config` | Checkpoint loading, the KV cache, and admission and batching |
| `device_config`, `moe_config` | Parallel configuration without a layout, and product transport; expert parallelism |
| `cuda_graph_config`, `worker_config` | CUDA graph capture, and the main processes and their workers |
| `update_config` | How a change is tried before it commits by itself, and when it rolls back |
| `logging_config` | Request and statistics logs |
| `chat_config`, `image_fetch_config` | Chat models: chat template, reasoning parser and image URL fetching |
| `video_config`, `media_config`, `video_job_config` | Video models: output sizes, durations, prompt capacities, condition media and asynchronous jobs |

A chat option or table is refused at startup when the served checkpoint is a video model, and a video table when it is a chat model. `GET /admin/policy` returns the current revision's package, its `config.toml` stating the options that differ from their defaults; `--log-level` and `--log-level-http` set the process's log levels. [Configuration](docs/configuration.md) describes every option, the policy package, its revisions and the layout format.

On a GPU, startup captures CUDA graphs for decode steps and for prefill steps before the server reports ready, and serving replays them. Prefill graphs hold up to `max_num_batched_tokens` prompt tokens and up to 31 prompts; decode graphs hold up to 128 requests. Both hold at most `max_running_requests` and at most as many requests as the KV pool holds a page of every cache group for. A prefill or decode step that no captured graph holds runs eagerly. Larger token budgets capture more graphs, so startup takes longer. Startup also runs every image encoder and decoder once, so the `uniserve-kernel-table` line it logs names the kernel of every call the server makes. With `attention_backend = "torch"` every call serves without graphs, since its captured calls would attend over every row's whole capacity.

[Configuration](docs/configuration.md) lists every option.

## Reproduce the measurements

The serving evaluator, `uniserve-eval` (`tools/eval`), runs HTTP workloads against a server, validates every response, and writes reproducible result bundles. It holds no model or campaign: each run is given a profile that states its servers and points, written at use. The [evaluation guide](docs/minimax_h3/evaluation.md) writes the profiles of the UniServe FastH3 post and the H200 measurements in the FastH3 guide; for example, the four-GB200 BF16 points:

```bash
cargo build --release -p eval
# The shell functions of the guide's "Writing a run", then its FastH3 block, write
# artifacts/minimax_h3/fasth3-gb200-4-bf16.toml.
target/release/uniserve-eval plan --config artifacts/minimax_h3/fasth3-gb200-4-bf16.toml gb200-4-bf16
target/release/uniserve-eval run --config artifacts/minimax_h3/fasth3-gb200-4-bf16.toml gb200-4-bf16
```

The guide states every campaign's points and the two-host procedure.

## Text and expert parallelism

For data-parallel serving, run one scheduler per GPU.

Adding `moe_config.expert_parallel = true` keeps the four schedulers' attention data-parallel and shards each expert layer across them, so every GPU holds a quarter of the experts and the schedulers exchange tokens at each expert layer. Every expert layer then runs in steps the schedulers take together: a scheduler without work joins each step another one starts, and all of them pad a step to the largest one's captured graph. With an NVFP4 checkpoint, `moe_config.exchange = "megamoe"` fuses each expert layer's exchange and expert computation into one kernel.

For text-to-video-and-audio generation with the FastH3 checkpoints, including the packed NVFP4 releases, use the [FastH3 guide](docs/minimax_h3/minimax_h3.md).

## Development and verification

The `justfile` exposes the canonical repository checks:

```bash
just lint
just test-rust
just test-python-fast
just test-python-gpu
```

`test-python-fast` runs without a CUDA device. The HTTP tests under `tests/python/e2e` start real servers on the host's GPUs with the `uniserve` the environment installed; `test-python-gpu` runs them on one worker per GPU, each case on the GPUs it leases, and a plain `pytest` deselects them (`-m "e2e and gpu"` selects them). The cases serve released checkpoints by Hugging Face repository and pinned revision from the standard cache; [Testing](docs/testing.md) lists them and the tests' options. A recipe's further arguments go to pytest; the FastH3 tests, for example, load a complete checkpoint on four GPUs:

```bash
just test-python-gpu tests/python/e2e/test_h3_parallel_http.py
```

The serving evaluator (`tools/eval`, `cargo build --release -p eval`) runs HTTP workloads against serving models, measures performance, validates response correctness, and writes reproducible result bundles. A profile names the servers a run starts (`[servers.<name>]`: the command, its environment and the address it serves; with `ready_fd = true`, as a UniServe server states it, the evaluator appends `--ready-fd N` to the command and takes the server as ready once its readiness record states that it serves, and the record's admin endpoint is the socket through which the run observes and changes the deployment) and the points it measures (`[benchmarks.<name>]`: the server, the task, the dataset and its source, the load, the sampling and the metrics each point protects), and `[suites.<name>]` lists points measured in order. Values may name environment variables as `${NAME}` or `${NAME:-default}`. A text point over ShareGPT at 16 requests per second, on a server of the policy package `PACKAGE`, whose `config.toml` serves the checkpoint `MODEL` (`model_dir`) as `Qwen3-30B-A3B` at `admin_config = { host = "127.0.0.1", port = 18081 }`:

```toml
[servers.qwen]
host = "127.0.0.1"
port = 18081
ready_fd = true
command = ["uniserve", "serve", "${PACKAGE}"]

[benchmarks.qwen-sharegpt-r16]
server = "qwen"
task = "text"
model = "Qwen3-30B-A3B"
dataset = "sharegpt"
dataset_revision = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
tokenizer = "${MODEL}"
load = { num_prompts = 200, request_rate = 16.0, warmup_requests = 1, seed = 42 }
sampling = { temperature = 0.0, sampling_seed = 42, ignore_eos = true, stream = true }
metrics = { output_throughput = "higher", mean_ttft_ms = "lower" }
```

`uniserve-eval plan --config <profile> <point or suite>` prints what a run would start and send; `run` measures it into `<output root>/<point>`. For each server with an admin socket, the summary's `server_metrics` states how far each counter of the deployment's observation store advanced over the measured window (a histogram's observation count and sum as `_count` and `_sum`), from its totals read with `/admin/query` just before the window opened and after it closed, which `server_metrics.json` keeps by series (`name{label="value",...}`).

For decision-session replay, give every row of a JSON Lines workload a non-empty `session_id`. The evaluator preserves file order within each session and sends the next decision only after that session's previous response. Set `load.request_rate = inf`; the sessions run concurrently, one request in flight per session. A session trace has no repeated-row warmup: each initial cold request is included in the measured trace. Session identifiers are benchmark metadata and are not sent to the model. Rows without session identifiers retain the seeded shuffled workload behavior.

A point whose `load.duration_s` is set runs a timeline load: it issues requests from the load's start for that many seconds, request `n` taking row `n mod num_prompts`, as a closed loop of `max_concurrency` streams when `request_rate = inf`, and then waits for every stream to end. Its `[[benchmarks.<name>.changes]]` act on the deployment while the load runs, each `at_s` seconds after the start or `delay_s` after an earlier change named by `after` (counted, with `phase`, from the deployment's event that reports the update entering that phase, a `log` event of source `update`): a `policy`, the directory of the policy package the deployment is to serve, applied whole with `PUT /admin/policy` (`force = true` applies it with `?force=true`) through the admin socket of a single-process `ready_fd` server, a `signal` to the one process of the deployment whose command line matches `process`, or a `command` run in the deployment's environment. Issuing continues while a change runs and for `load.settle_s` after the last one ended. Each record of `requests.jsonl` holds its chunks' arrival times and, when the request asks for them (`sampling.extra_body` with `logprobs`, `top_logprobs`, `return_token_ids` and `return_tokens_as_token_ids`), its token ids and their candidates; `timeline.json` holds the load's clock, each change's answer with its times and the deployment's `log`, `process` and `revision` events, which the run reads through the store's subscription (`/admin/query` with `Accept: text/event-stream`). `uniserve-eval check-change <run> --control <control> --check-metric <counter> --replay <verdict.json>` checks such a run against the same load without changes: no request failed or lost its connection, each stream carried exactly its requested tokens and agrees with the control's output of its row by the near-tie rule of vLLM's `check_logprobs_close`, each request submitted during the change received its first token by the change's end plus the control's 99th-percentile time to first token, the named check-failure counter's total in the deployment's store reads 0 after the window and the replay verdict is `pass`. It writes `result.md` with these items and the longest inter-token gap and time to first token beside the control's, and prints them as one JSON line.

A change may also be a `signal` with a `host`, sent to the one process of this user on that host whose command line matches `process`, found and signalled there over `ssh` in batch mode; or a `cancel` of that many streams of a closed loop at once, the earliest issued or with `cancel_order = "newest"` the latest issued, which the client disconnects mid-stream: their requests are recorded with the classifier `cancelled`, the streams go on, and the point leaves them out of its metrics and validation (`cancelled_count`) and `check-change` out of its items. A change that moves the deployment to another configuration names it (`configuration`, a rollback naming the one it returns to), and each `--control-of NAME=PATH` gives `check-change` the control of a configuration, started in it: a request agrees with the control of the configuration it was submitted on or of one a change overlapping its life moved to, since a request in flight across an epoch's switch may end on either side. `--switch NAME --block-size N` adds a check of the cache a switch carries over: the first request submitted after the switch named `NAME` ended, and before the next change, of each prompt that received a token before the switch, reports as `usage.prompt_tokens_details.cached_tokens` at least the whole blocks of its prompt less the last token, which an admission always computes. `uniserve-eval check-faults <run> --reference <alone> --check-metric <counter> --replay <verdict.json>` checks a run whose changes are faults against its rows served alone, the bundle of a point with `max_concurrency = 1` and no `duration_s`, which sends each row once with nothing else running: every request not cancelled answered 200 with its requested tokens, which agree with its row's by the near-tie rule, and the invariants' evidence holds. A timeline load also keeps every counter and gauge row each server's store bucketed from its start on, into `stats.jsonl`: one row per second, metric and label set (a counter's increase in that second, a gauge's latest reading), with its origin. `uniserve-eval check-stop <run> --control <control> --headroom-tokens N` checks a run in which a process of the deployment was stopped with `SIGSTOP` and continued with `SIGCONT` against the same load without the signals: no request failed; in each stop window the gaps between tokens that ended in it have their p50 and p99 within the control's gaps of the same window, and no part of it without a token outlasts the control's longest gap; the counter of calls left out for late blocks (`uniserve_scheduler_calls_without_blocks_total`) rises only in the seconds a stop window explains, after some stream decoded `N` tokens in it, and every request in flight at a window's end completes.

## Repository layout

The [architecture guide](docs/architecture.md) describes how the daemons, the engine process and its workers divide the work, how a main process's threads divide it, and how a request travels through them.

```text
src/                    The root package, `uniserve`: the command line, logging, the `serve` command, and `main.rs`, which dispatches every command; the base executable holds the daemon, the Storage Server and the Build Server, and links no engine or model
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
crates/build-server/  The Build Server: builds packages into modules, keeps them and hands them out; one per deployment and node
crates/testkit/         Test doubles at the engine's seams: the executor of a worker group (`FakeExecutor`), the Storage Server's buffer pool in process (`InProcessPool`) and a fake engine over both; never in a serving build
crates/manager/         `uniserve-manager`: an optional multi-node manager, a client of each node's `/admin`
tools/eval/             `uniserve-eval`, the serving evaluator
tools/profile/          `uniserve-profile`, the reader of profiling sessions and of Nsight Systems and cProfile captures
tools/verification/     The buffer pool's design model and its checks
configs/                Deployment files: H3 (`minimax_h3/`) and dedicated experts (`afd/`)
docs/architecture.md    Processes, the rank planes and the request path
docs/configuration.md   The serving option tree, configuration files and deployment files
docs/admin.md           The admin API: a node's status, event log, deployment and configuration
docs/profiling.md       Timeline and native profiling
docs/minimax_h3/        H3 serving and evaluation guides
docs/qwen3_moe/         Qwen3-MoE serving and its GSM8K quality confirmation
```

## License

Apache-2.0.
