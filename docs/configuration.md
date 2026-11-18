# Configuration

A UniServe deployment is configured by its policy package and nothing else. The package's `config.toml` is one tree of serving options: its top level holds the options most deployments state, the rest sit in `*_config` tables, one per part of the system they configure, and `[[worker_groups]]` tables state the [deployment layout](#deployment-layout) of a parallel configuration other than one tensor-parallel worker group per scheduler.

## Policy package

A policy package is a directory: `config.toml`, required, and `programs/`, optional, the sources of the programs its configuration lists. The directory's name is the deployment's name: it names the deployment's persistent state on every node ([Revisions](#revisions)). `uniserve serve` takes the package and no option of the configuration:

```bash
uniserve serve configs/minimax_h3/ulysses4
```

```toml
# configs/minimax_h3/ulysses4/config.toml, with the options of a four-GPU host added
model = "FastVideo/FastVideo-FastH3-8-Step-V2"
served_model_name = "FastH3"
max_running_requests = 2
max_model_len = 10240

[admin_config]
host = "0.0.0.0"
port = 8000

[video_config]
text_capacities = [1024]

[[worker_groups]]
id = "model"
ranks = ["cuda:0-3"]
# ...
```

An option the file does not state takes its default, and `GET /admin/policy` returns the file of the current revision with the options it states, defaults left out, so the file read back starts the same deployment and does not read options of another model family or rank-assignment mode as stated. A start and a change are one operation: a start applies the package to an empty deployment, so every option is a change, and every option a start can state a change can change ([admin API](admin.md#policy)).

`uniserve serve PACKAGE` takes two more arguments, neither an option of the configuration. `--ready-fd FD` names an inherited descriptor the daemon writes its readiness record to: one JSON line once its listeners are bound and one once it serves, each stating the endpoints it bound ([admin API](admin.md#listeners)). `--log-level` sets the log level of every target of the `uniserve` process, and `--log-level-http` the level of the daemon's HTTP stack (the server and the media-fetching client); every subcommand takes them, before or after its name. A level is one of tracing's names (`off`, `error`, `warn`, `info`, `debug`, `trace`), `info` when unset, and any other value is refused. The environment sets no level (`RUST_LOG` is not read). The daemon starts its engine processes at its own `--log-level`, and a Storage Server its helpers at its own; lines name the level as tracing does (`WARN`, not `WARNING`).

Every option is checked by its type when the package is read, and an error names its path: an unknown option, a value its option does not take, or zero for a positive count. A value is written as TOML writes it: `2`, `true`, `"text"`, `[1024, 4096]`, `{ mode = "quality" }`. A duration is whole seconds or text with units (`"500ms"`, `"30s"`, `"5m"`, `"1h"`). A byte size is whole bytes or text of an integer with a decimal (`"20MB"`) or binary (`"512MiB"`) unit. Paths in options resolve against the server's working directory. A package is checked the same way whether it starts the deployment or changes it: a target is refused while the deployment serves exactly when a start with it would be.

### Nodes

A deployment's nodes are its `hosts`, by host name, the leader first. Every node runs `uniserve serve` with the same package and finds its place in the list by its own host name (`uname -n`); the leader serves the API, applies every package and runs the engines' control, and every other node joins the leader at the first host's `admin_config.join` port (`29700` by default) and supervises the processes the layout places on its host. Each host resolves the leader's name. A package that lists no hosts is one node on whichever host serves it, and a host that does not find itself in a package that lists hosts refuses to serve it, naming the hosts. A rank whose `ranks` entry names no host runs on the leader's host.

```toml
hosts = ["gb200-a", "gb200-b"]

[[worker_groups]]
id = "model"
ranks = ["gb200-a/cuda:0-3", "gb200-b/cuda:0-3"]
```

A node other than the leader reads its own package's `hosts`, `admin_config.join`, `build_config` and `storage_config`; the rest of the deployment is the leader's. The packages of the hosts state the same `hosts` and join port, and each states its own host's Build Server.

### Revisions

Every package a deployment converged to is a revision, numbered upward and kept as a complete package in the deployment's persistent state directory on each node, `$XDG_STATE_HOME/uniserve/deployments/<name>` (`~/.local/state/uniserve/deployments/<name>` without `XDG_STATE_HOME`), never in the runtime directory, which ends with the user's session:

```text
<state>/
  revisions/
    41/  config.toml  programs/  lock.toml  revision.toml
    42/  config.toml  programs/  lock.toml  revision.toml
  current -> revisions/42
  program-state/
```

`config.toml` and `programs/` are the package, what `GET /admin/policy?revision=N` returns. `lock.toml` holds the module revisions the apply fixed at the Build Server, so a revision named `latest` is resolved once, when it is applied, and never again on a restart. `revision.toml` records the change: when it began (`time`), who wrote it (`writer`: `start` for the package a node started with, `operator` for a `PUT /admin/policy`), each option it changed with its effect and its value before and after (`changes`), its heaviest effect (`effect`), how it ended (`result`: `converging`, `converged`, `rolled_back` or `abandoned`), why it did not converge (`reason`) and how long it took (`seconds`). A change writes its revision's directory and syncs it before it executes, and replaces `current` with a link to it in one rename once the deployment converged; a change that did not converge records why and leaves `current` where it was. A revision a leader left converging when it ended is recorded `abandoned` when the deployment starts again. The newest 64 revisions are kept beside `current` and the one that converges.

A node started with the package its current revision holds starts that revision with the modules its `lock.toml` fixed. A node started with another package starts it as a change of the current revision: it becomes current once the deployment serves, and the current revision starts instead when it does not come up.

### Effects

Every option declares how a change of it takes effect, and a change takes the least its options need: the effects of the options it changes, merged per component.

| Effect | What happens |
| --- | --- |
| `step` | Each scheduler reads the new value at its next step; an operation the daemon applies itself (`paused`, `asleep`, `flush`, `adapters`, `served_model_name`, `retire`, `profiling_config.session`) takes effect as the [admin API](admin.md#operations) states |
| `order` | The experts' new order is prepared beside the batches and committed at one computation boundary |
| `program` | The program, or the scheduling policy's arguments, is replaced between two of its calls, as an update on trial |
| `generation` | The serving engines prepare the new kernel modules, capture their graphs in the background and switch to them, as an update on trial |
| `storage` | The deployment's Storage Server on each node is replaced, and the engines follow as an epoch |
| `epoch` | A new epoch's engines start beside the serving ones, take over its requests, and the previous epoch drains, as an update on trial |
| `daemon` | The daemon is replaced by a successor in a hot restart, which applies the rest of the change |

An update on trial commits by itself once it served for `update_config.trial` with no fault and no threshold rolling it back ([`update_config`](#update_config)); its change converges then. On an engine, an epoch includes every lighter effect, and the daemon restarts first, its successor applying the rest. The option tables below name each option's owner: what builds a main process takes an epoch, a module selection a generation, a package parameter the policy's program, and an option of the daemon's own a hot restart.

### A deployment's servers

Each deployment runs one Storage Server and one Build Server on each node it serves on, beside its daemon there; no server is shared between deployments, and each GPU belongs to one deployment. The servers keep their sockets, locks, readiness records and logs in the deployment's runtime directory, `$XDG_RUNTIME_DIR/uniserve/deployments/<name>/`, the name being the policy package's directory: `storage-server.sock` and `build-server.sock`, with `.ready`, `.lock`, `.start` and `.log` beside each. An operator's command names the server it reaches by `--deployment NAME`, or by `--socket PATH`.

Before a node's daemon starts an engine there, it claims the engine's GPUs for the deployment in the node's GPU registration file, `uniserve-gpus` in the system's temporary directory, one for every deployment of every user of the node: a GPU another deployment holds refuses the engine's start with an error naming the holding deployment, its user and its daemon's process. A claim lasts as long as the daemon that holds it (a hot restart's successor goes on holding it) and ends when that process ends, however it ends. GPUs are named by their node ordinal: the CUDA ordinal the deployment places a rank on, read through `CUDA_VISIBLE_DEVICES` when the daemon's environment narrows its devices, which then names them by ordinal.

The deployment's Storage Server (`uniserve storage-server serve`, see `architecture.md`) serves with the settings of one document (`storage::Settings`, TOML), which the deployment states (`storage_config` below, with the connector of `kv_cache_config.mooncake`): the node's daemon starts the server with them when the node has none (`storage-server serve --daemon --socket PATH --settings -`, the document on its standard input), and keeps a server of the deployment that serves with others serving with them: it takes the settings it reads when it acts at once (`Request::Configure`), and is restarted in place with the others (below). An operator starts one by hand with `uniserve storage-server serve --deployment NAME [--settings FILE] [--ready-fd FD] [--daemon]`: the document's `[server]` table holds `storage_config`'s keys and its `[connector]` table `kv_cache_config.mooncake`'s, every key at its default when unstated, and a key no table knows is refused. The server binds its socket and its TCP endpoint before anything else (port `0` allowed) and reports both in its readiness record (`storage`, `peer`) with its settings: on `--ready-fd`, or, with `--daemon`, on its starter's standard output once it serves. `uniserve storage-server status` prints that record from the running server, and `uniserve storage-server stop` stops it, which releases everything it holds, and returns once it ended. `uniserve storage-server list` prints what it holds, with the deployment's buffer pool on lines led by `buffer pool of NAME:` (the blocks of each region and class that are shared, private or retired, each worker group's last completed batch and the stamps its deferred releases, commits, transfers and copy ends wait for, and the requests whose copies are in flight), or with `--json` the same listing as one JSON document (`storage::protocol::Listing`) for tools. The deployment's storage is named by the deployment's name, so a server restarted in place finds the storage it left under it. A deployment's host KV cache is the host regions of its buffer pool, one memfd per rank index, which the server admits against the host's storage when a session first states the region. A deployment whose package states `retire = true` stops its Storage Server on the leader's node when it comes down at its stop signal, which releases its weights and caches ([admin API](admin.md#nodes)).

The deployment's Build Server (`uniserve build-server serve --deployment NAME`, see `architecture.md`) is configured by `build_config`, whose keys are the flags of its `serve` command: `package_root` and `store_roots` (`build-server/packages` and `build-server/modules` in the deployment's data directory, `$XDG_DATA_HOME/uniserve/deployments/<name>`, else `~/.local/share/uniserve/deployments/<name>`; the first store root is written, the rest are read-only roots), `upstream` (the socket of the upstream Build Server that accepts `install`, on a host that is not it), `source` (the UniServe source tree the server builds from: uv creates the server's Python environment from the tree's locked project `crates/build-server/python` when the server starts, and the upstream server installs the tree's serving packages, every kernel package of `crates/ops/kernels`, the release's programs (the crates with a `package.toml` under `crates/scheduling/policies`, which a package names as `uniserve:<name>`) and the engine, before it serves, each as a new revision only when its build inputs changed; unset, the server has no Python environment, installs nothing by itself and refuses a policy package's own program crates, which it builds in the source tree's workspace), `read_only_stores` (store directories this server serves and never writes, each as `build --store` makes one with its `packages` and `modules`: the store a release ships with the modules of its own packages, whose revisions and modules a host serves with no toolchain, so a first start builds nothing; a revision the host installs is numbered after them), `targets` (the GPU architectures it builds for, the host's devices' unless stated; a target is one the Build Server's target description `crates/build-server/python/src/uniserve_build_server/targets.toml` states), `max_jobs` (the builds that run at once, and the compiler processes each build runs, 4 unless stated), `host_reserve` (the host memory in bytes a build leaves free, 8 GiB unless stated) and `peak_memory` (a table of toolchain kind to the bytes one build of it reserves: `triton` 8 GiB, `cute_dsl` 10 GiB, `nvcc` 4 GiB and `cargo` the larger of `max_jobs` × 2 GiB and 16 GiB unless stated; `--peak-memory KIND=BYTES` on the command line). No option names a toolchain: the server runs its Python environment's program `uniserve-build-server toolchain --json` once when it starts (`crates/build-server/python/README.md`), which describes the Triton and CuTe DSL compilers the environment locks, the CUDA toolkit of the release its lock fixes (found through the `cudart-<release>` pkg-config module NVIDIA's packages register) and the host C++ compiler; a host without the environment refuses those packages with the reason. cargo is rustup's, at the release the workspace's `rust-toolchain.toml` pins. The server keeps what its builds share in `build-server` of the deployment's cache directory (`$XDG_CACHE_HOME/uniserve/deployments/<name>`, else `~/.cache/uniserve/deployments/<name>`): the pinned third-party sources, cargo's target directory, the vendored registry sources of each set of external packages a lockfile locks, the copy of the source tree's workspace a policy package's program crates are built in, and the builds' workspaces. Deployments therefore build and keep their products apart; the read-only stores a deployment names are read alike by every deployment that names them. The daemon prepares an epoch's modules at the deployment's Build Server on its node, and starts it with these settings when none serves there. A Build Server publishes the settings it serves with in its readiness record (`build-server.ready` beside its socket); a Build Server of the deployment that serves with other settings, one an earlier daemon of the deployment started, is stopped and started again with the deployment's. `uniserve build-server install --deployment NAME DIR` installs a package on the upstream Build Server and `uniserve build-server list --deployment NAME` prints the installed packages, revisions and modules, with the git commit each revision was installed from and whether its tree was modified, and `uniserve build-server resolve --deployment NAME OP BACKEND REVISION` prints where the store holds one module's files, building the module first when it is missing. `uniserve build-server build --keys FILE [--keys FILE ...] --packages DIR --target SM --store DIR [--source DIR] [--max-jobs N] [--peak-memory KIND=BYTES]` builds a store with a Build Server of its own, in process, which compiles with the Python environment of the source tree `--source` names, which holds the store's package root while it runs: for every `operator<TAB>descriptor key` line of the key lists it installs the operator's packages from the package tree `--packages` (one directory per operator, one package directory per backend) into `DIR/packages`, builds the key for the target into `DIR/modules` with at most `--max-jobs` builds at once (4 unless stated), prints `ok` or `FAILED` per key, and fails with the keys that did not build. A package whose build inputs are unchanged since its latest revision in the store keeps that revision, so building more keys into an existing store extends it; a change of the package's files, of the toolchain's version, or of the Build Server environment's compile driver, pins or compilers (`uniserve_build_server`) installs a new revision whose modules are built again. Every module is stored under the digest of the inputs it was built from (`<op>/<backend>/<revision>/<inputs>/<target>/<key>`), so a module is never served for other inputs, whichever package root numbered its revision; every installed revision records its inputs, and a package root holding a revision without its record is refused. `DIR/packages` and `DIR/modules` are then a `serve`'s `--package-root` and `--store-root`.

A store is built for a deployment from its kernel inventory. Every kernel a serving deployment's main processes launch is counted in its observation store as `uniserve_kernel_launches_total`, labelled by its operator (`op`) and its descriptor key (`descriptor`) ([admin](admin.md)), so the inventory is one query of `/admin/query`. Each row of the answer, written as one `operator<TAB>descriptor key` line, is the key list `build-server build --keys` reads; a store built from it holds the modules of every kernel the deployment launched, for the target it is built for:

```bash
curl --unix-socket "$admin" -G http://admin/admin/query --data-urlencode "q=
  SELECT labels['op'] AS op, labels['descriptor'] AS descriptor, sum(value) AS launches
  FROM stats WHERE name = 'uniserve_kernel_launches_total' GROUP BY 1, 2 ORDER BY 1, 2" \
  | jq -r '.[] | "\(.op)\t\(.descriptor)"' > keys.txt
uniserve build-server build --keys keys.txt --packages crates/ops/kernels --target sm_100a \
  --source . --store store
```

`build --package DIR` installs a package that fills a whole interface with one module, the engine (`crates/engine`) or a scheduling policy, and builds that module for its own target (the host's Rust target triple, or `wasm32-unknown-unknown`) with cargo; it is repeatable and combines with `--keys`.

Some options configure one kind of model, by what the served model family serves: `max_num_batched_tokens`, `scheduler_config.chunked_prefill_size`, `speculative_config`, `chat_config` and `image_fetch_config` configure chat models (Qwen3-MoE); `video_config`, `media_config` and `video_job_config` configure video generation (MiniMax-H3). Startup refuses such an option stated for a model of the other kind, naming the option and the served family.

## Options

Each option's destination follows what its value is used for: a value that builds the main process is **build config**; one that selects an implementation is a **module selection**; one that only steers a decision is a **package parameter**; what the daemon reads stays in the **host layer**. An option a section does not name separately has the destination stated under that section's heading; the sections whose options split state it per field below.

Three backend-name fields are read with `attention_backend` (`modules_of`, `crates/daemon/src/epoch.rs`): `prefill_attention_backend` and `decode_attention_backend` each override it for their phase, and `moe_runner_backend` selects the grouped expert GEMM. A backend name selects the package of that name at the Build Server for every descriptor key it covers; a key it does not cover, and every key of an operator whose backend is unset, takes the installed package that covers it with the highest manifest `priority` (`architecture.md`, the Build Server). A named backend no installed package has is refused when the epoch's modules are prepared.

### Top level

| Destination | Options |
| --- | --- |
| Operations the daemon applies in place (`step`, [admin API](admin.md#operations)) | `served_model_name`, `paused`, `asleep`, `flush`, `retire`, `adapters` |
| Build config | `model`, `model_dir`, `max_model_len`, `lora`, `lora_dir`, `dtype`, `quantization`, `max_running_requests`, `mem_fraction_static`, `tensor_parallel_size`, `data_parallel_size`, `worker_groups`, `hosts`, `engine_host` |
| Host layer | `mm_processor_kwargs` |
| Module selection | `attention_backend` (attention), with `prefill_attention_backend` and `decode_attention_backend` overriding it per phase |
| Package parameter | `max_num_batched_tokens` (the scheduling policy's bound); read at `crates/daemon/src/epoch.rs:1364-1366` |
| Retired, refused by name | `quantization_config`: the option is stated as `quantization`, and the retired spelling is refused with the name to state instead |

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | Required, or `model_dir` | The Hugging Face checkpoint the deployment serves, `<owner>/<name>[@<revision>][/<folder>]`: a repository at its default branch or at a branch, tag or commit, and the folder of it that holds the checkpoint when it is not the root. The leader resolves the revision to its commit and reads the checkpoint from the standard Hugging Face cache's snapshot of that commit (`$HF_HOME/hub`, `~/.cache/huggingface/hub` by default), downloading what the snapshot lacks: a chat model's every file, or a diffusers pipeline's root files and the folders of the stages its index declares. `HF_ENDPOINT` names a mirror and `HF_TOKEN` the access token (else the token `hf auth login` wrote); with `HF_HUB_OFFLINE=1` nothing is downloaded, and the cache must hold the revision's snapshot. The engines open the snapshot, so every node of the deployment reads the same cache path |
| `model_dir` | Unset | A local checkpoint directory the deployment serves, read in place, instead of a repository; stating both is refused |
| `served_model_name` | `model` or `model_dir` | The single model name the API serves; a change renames the API from the next request on: a request naming the former name is refused, and the requests in flight under it end as they would have |
| `max_model_len` | The context length the checkpoint states where its family keeps it (a Qwen3 checkpoint's `max_position_embeddings`, a Qwen3.6 or Qwen3.8 Flash Next checkpoint's `text_config.max_position_embeddings`); a checkpoint that states none is refused, and a media pipeline (MiniMax-H3) has none, so its deployment must state it | Longest sequence a request may reach, in tokens; for a media pipeline the longest prompt presentation it admits |
| `lora` | Unset | A weight adapter fused into the checkpoint's weights at load and serving every request: the Hugging Face repository, or the folder of one, `<owner>/<name>[@<revision>][/<folder>]`, resolved as `model` is, that holds the adapter in its family's declared file and the contract of the student the adapter is, which states the network it edits, its schedule and its attention; video families only, and an adapter without a contract is refused (MiniMax-H3: `adapter_model.safetensors` and `fastvideo_inference.json`, [FastH3 adapters](minimax_h3/minimax_h3.md#fasth3-adapters)) |
| `lora_dir` | Unset | The adapter as a local directory holding the same two files, read in place, instead of a repository; stating both is refused |
| `dtype` | `"bfloat16"` | Element type of parameters and activations: `bfloat16`, `float16` or `float32` |
| `quantization` | `{}` | Precision policy: `mode` (`quality`, `balanced`, `performance`, `maximum`) and representations selected by module under `modules` (`bf16`, `fp16`, `fp8`, `nvfp4`); empty selects the checkpoint's own quantization. The model family interprets it when the engine opens the checkpoint and refuses a policy it does not serve, naming each module: MiniMax-H3 serves every mode and selection but `video_vae = "bf16"` over a dense checkpoint, converting its FP8 and NVFP4 modules' projections at load and measuring each activation's scale on every call ([FastH3 precision](minimax_h3/minimax_h3.md#precision)); the chat families state no mode and no module, and a checkpoint that declares its own quantization takes no policy |
| `mm_processor_kwargs` | `{}` | Options of the model's multimodal processor; MiniMax-H3 takes `reference_image_short_edge` and `enlarge_references` ([FastH3 guide](minimax_h3/minimax_h3.md#startup-and-capacity)) |
| `attention_backend` | Unset | Attention package: unset, the Build Server selects, for each attention descriptor key the deployment launches, the installed package that covers it by manifest priority; a package's backend name (`trtllm`, `triton`) serves every key it covers, and the others are selected as when it is unset |
| `max_running_requests` | `128` | Concurrently resident requests, clamped to what the workers hold: each worker group holds this many request slots, or as many as its ranks' shares of their GPUs hold beside the weights, the graph pools and the fewest KV units when fewer fit (a slot holds a token request's slot state and, for a model that keeps a recurrent state, one checkpoint of it, or a video request's trajectory and its products on the rank), and a start whose share holds no slot fails, naming the bytes one slot needs |
| `max_num_batched_tokens` | The widest prefill batch the workers' captured graphs hold; `8192` when they capture none | Token work admitted in one scheduling step, the scheduling policy's step budget when its package states none; token models only |
| `mem_fraction_static` | `0.7` | The deployment's share of each device's storage for weights, static allocations, graph pools and the KV cache, in (0, 1]: the limit `⌊total × fraction⌋` of the GPU's quota, which both epochs of a configuration change and the host's Transfer Server share (`docs/admin.md`); a deployment group's `mem_fraction_static` overrides it for the group's ranks, and a GPU that holds ranks of several groups has the shares of them all (below) |
| `tensor_parallel_size` | `1` | Tensor-parallel workers of each scheduler's group without `worker_groups` |
| `data_parallel_size` | `1` | DP ranks, each a scheduler with its own KV cache and worker groups. Independent replicas each run an engine of their own, and the leader sends a request to the one with the fewest in flight. With `moe_config.expert_parallel`, replicas of one rank each (`tensor_parallel_size` 1) share their experts and exchange tokens, so one engine runs every replica's scheduler and the request names the scheduler the leader chose for it; replicas of several tensor-parallel ranks each shard their experts over their own ranks and stay independent replicas. With `worker_groups`, the file's groups that serve requests form the schedulers as equal consecutive blocks |
| `worker_groups` | Unset | The [deployment layout](#deployment-layout): `[[worker_groups]]` tables placing worker groups and their blocks |
| `hosts` | `[]` | The deployment's nodes, by host name (`uname -n`), the leader first ([Nodes](#nodes)); empty, the deployment is one node on whichever host serves it |
| `engine_host` | The leader's host | The host of `hosts` that runs the engine and any ranks without an explicit host. Changing it starts a successor epoch there while the daemon that serves the API remains on the leader |
| `paused` | `false` | Whether new requests wait in the daemon instead of reaching an engine; the requests in flight go on. A change takes effect from the next request on |
| `asleep` | `false` | Whether the engines give back their KV cache and CUDA graph storage and keep their weights. While asleep new requests wait; once awake again, the token requests that were in flight are recomputed and the video requests continue from their stored step. A change takes effect at the engines' next step boundary |
| `flush` | `0` | How many times the prefix cache was flushed: each change of the number revokes every cached prefix page at the engines' next step boundary, the requests in flight keeping the pages they hold |
| `retire` | `false` | Whether the deployment's storage (its weights and caches) is released at its Storage Server once the deployment comes down at its stop signal; otherwise it stays retained for a start of the same deployment. Read when the deployment comes down |
| `adapters` | `{}` | The LoRA adapters the workers hold (`[adapters]`), each the directory of a PEFT adapter under the name a request states as its `model` to compute with it; a change loads the adapters it adds into free slots of the workers' adapter pool and unloads the ones it removes, from the next request on, and an engine that starts later loads every one of them ([admin API](admin.md#lora-adapters)) |

A checkpoint that declares its own quantization in `config.json`'s `quantization_config` is read in the representation each weight's declaration and shard state; the declaration's `quant_method` names its producer, and the producer served is ModelOpt (`modelopt`), read in its own fields (`quant_algo`, `quantized_layers`). NVFP4 (`quant_algo: NVFP4`) runs NVFP4 products over activations encoded to NVFP4. A `MIXED_PRECISION` export states one algorithm per module in `quantized_layers`, and each module is read in its own: `FP8` is a per-tensor E4M3 product whose activation is encoded at the calibrated static `input_scale`; `NVFP4` as above. A representation no operator of this build multiplies is refused at load, naming the module, and never decoded into a dense matrix in its place: ModelOpt's weight-only `W4A16_NVFP4` and `FP8_PB_WO`, another producer's declaration (such as `quant_method: fp8` with a `weight_block_size`), and a shard that stores a `weight_scale_inv` beside a weight. This refuses Qwen3.8-Flash-Next-FP8, the Qwen3.8-Flash-Next NVFP4 export of NVIDIA (whose drafter's experts are `FP8_PB_WO`) and Qwen3.6-35B-A3B-NVFP4 (whose experts are `W4A16_NVFP4`) until those operators exist. A packed ModelOpt export of a diffusers pipeline declares its quantization in `modelopt_manifest.json` at its root instead: its calibrated projections are stored as `weight_packed`, `weight_scale` and `weight_tensor_scale`, run the same NVFP4 products, and encode their activations at the tensor scale the manifest states for each module (`activation_amax / (6 · 448)`). The key/value cache keeps the element type `kv_cache_config` states, the activations' otherwise.

### `server_config`

**Destination: host layer.** The leader daemon's application server reads these: how the HTTP and OpenAI API answer; where the daemon listens is `admin_config`'s.


| Option | Default | Meaning |
| --- | --- | --- |
| `api_key` | Unset | Bearer token every route but `/health` and `/metrics` requires; the configuration is its one source, and a stated key is not empty. Never printed |
| `request_timeout` | Unset | Time until a response head is sent, answered with 504 when it expires; streamed bodies are not bounded |
| `max_concurrent_requests` | Unset | In-flight chat and synchronous video requests above which requests are shed with 503 |
| `shutdown_timeout` | `"30s"` | Longest drain of in-flight requests at shutdown; `0` stops at once |
| `request_id_headers` | `false` | Add an `X-Request-Id` response header with the ID the request ran under |
| `max_request_head` | `"64KiB"` | Largest request head (request line and headers), and largest trailer section of a chunked request body; a larger head is answered with 431 |
| `max_request_body` | `"64MiB"` | Largest request body of every route but the video submissions, which take what `media_config.max_request_bytes` of base64 media needs on top of it; a larger body is answered with 413 |
| `stream_keep_alive` | `"15s"` | Time a server-sent-events response waits for its next event before a comment (`:`) keeps the connection alive |
| `max_request_id` | `128` | Longest client `X-Request-Id`, in bytes, taken as the request's ID; a longer one is replaced by a generated ID |
| `retry_after` | `"1s"` | `Retry-After` of a request the concurrency limit sheds, in whole seconds |

Every response states its body's framing in its head: `Content-Length` when its length is known, chunked otherwise; an HTTP/1.0 client, which cannot read chunks, reads a body of unknown length to the end of its connection. A request body may be chunked, with chunk extensions and a trailer section, which the server reads and discards.

### `admin_config`

**Destination: the daemon's own options.** The daemon's listeners read it: the client API's address, the admin API's Unix socket, `admin.sock` in the daemon's instance directory, which it serves apart from the client API ([admin API](admin.md#listeners)), and the port the leader accepts the deployment's other nodes on. The daemon binds every listener when it starts and states the endpoints it bound in its readiness record.

| Option | Default | Meaning |
| --- | --- | --- |
| `host` | `"127.0.0.1"` | TCP address the daemon's client API listens on |
| `port` | `0` | TCP port the daemon's client API listens on; `0` picks a free one, which the readiness record states (`client`) |
| `uds` | Unset | Unix domain socket the client API listens on instead of TCP, owned through the lock `NAME.lock` beside it: a socket file a daemon that is gone left is replaced, a path another daemon owns is refused, and the file is removed at shutdown |
| `join` | `29700` | The TCP port the leader accepts the deployment's other nodes on, on every interface, and the port they join it at on the first host of `hosts`; bound only by a deployment of several nodes ([Nodes](#nodes)) |

### `storage_config`

**Destination: the deployment's Storage Server on each node.** The settings of the deployment's Storage Server on every node it serves on (`[server]` of the server's settings document): the node's daemon starts its server with them, and a change of them reaches the running servers. A deployment of several nodes (`hosts`) has its servers accept other hosts on `0.0.0.0:0` unless `tcp` states an endpoint.

A change takes effect at the next step (`step`) when the running servers take every changed setting, each the next time it acts on it: `host_reserve_gib` at the next admission of host storage, `transfer_chunk_us` at each copy thread's next chunk and `transfer_progress_ms` at each task's next report, `child_restart_s` and `child_restart_max_s` at a child's next start, `shoot_down_s` at each shoot-down's next check, and `transfer_deadline_s` and `put_deadline_s` for the tasks that start from then on while no `buffer_pool_trace` records. Otherwise the change replaces each node's server (`storage`): the server is restarted in place with the new settings, its successor going on with every connection, tensor and buffer pool, and the engines follow as an epoch. That is so for `tcp` (the listener for other hosts), `transfer_calibration` and `nixl_transports` (which the Transfer Server and its NIXL agent read once when they start), `buffer_pool_trace` (a trace starts from each pool's image, which states a pool faithfully only while none of its tasks is in flight, as at a restart in place) and, while a trace records, the two deadlines, which are part of each pool's state the trace holds only in an image; a change of `kv_cache_config.mooncake`, the connector the server runs as its child, restarts it as well. The leader has every node's server take the change before the engines follow, and gives the servers back the serving settings when the change does not converge.

| Option | Default | Meaning |
| --- | --- | --- |
| `tcp` | Unset | The TCP endpoint (`HOST:PORT`, port `0` picks one) on which the server also accepts other hosts' heads and peers, which its readiness record states as `peer`; local connections only when unset. The endpoint is not authenticated |
| `host_reserve_gib` | `10` | Host storage, in GiB, that new host storage (a buffer pool's host region, the host copy of an epoch's weights) leaves to the host's other processes: storage the host's available storage less the reserve cannot hold is refused |
| `transfer_calibration` | Unset | The host's transfer calibration, the JSON document `uniserve storage-server calibrate --output PATH` writes on each host the deployment serves on (below): the server reads it when it starts and refuses to start on one it cannot read or that does not hold together. Unset: the server runs no Transfer Server, and refuses transfers as unavailable |
| `transfer_chunk_us` | `1000` | How long one chunk of the Transfer Server's copies lasts at the link's bandwidth, in microseconds; a chunk lasts at least the link's calibrated per-copy cost |
| `transfer_progress_ms` | `100` | How often a running transfer task reports the bytes it copied at most, in milliseconds |
| `nixl_transports` | `"selected"` | The transports the Transfer Server's NIXL agent takes between hosts, over the host's usable RDMA ports ([NIXL transfers](nixl.md)): `"selected"`, those UCX selects by itself; `"rdma"`, RDMA between processes and CUDA copies within one, so that a READ between two processes of one host crosses the network adapter as one between hosts does |
| `child_restart_s` | `5` | Seconds the server waits before it starts a child process again (its Mooncake connector or its Transfer Server), doubled for each process in a row that ended before it connected |
| `child_restart_max_s` | `300` | The longest wait, in seconds, before a child process starts again |
| `transfer_deadline_s` | `60` | How long a buffer pool's transfer of a request between worker groups may take, in seconds, before it is cancelled and answered as failed |
| `put_deadline_s` | `60` | How long a buffer pool's put of host copies to the remote store may take, in seconds, before it is stopped and answered with the keys put so far |
| `shoot_down_s` | `60` | How long a process the server killed (a previous writer) or followed (an engine that is gone) may take, in seconds, to leave every GPU whose storage it held once it exited; past it the shoot-down fails, the GPU needs a reset, and the requests that waited for it fail |
| `buffer_pool_trace` | Unset | A directory the server records every buffer pool's inputs and completions in (`specs/buffer-pool.md` §18); a failure to write stops the trace, not the pool |

The Transfer Server copies by its host's calibration, which `uniserve storage-server calibrate --output PATH` measures on the host and `transfer_calibration` names (`crates/storage-server/src/transfer/calibration`). The calibration states the GPUs the Transfer Server copies on (with `--host-only`, none, and the Transfer Server loads no CUDA driver), the device storage its CUDA context takes on each GPU as NVML attributes it, and each path's bandwidth and per-copy cost: each GPU to itself, each pair of GPUs that can address each other, each GPU to and from host storage on its NUMA node, and host storage to itself; with `--reads`, NIXL READs into host storage and each GPU through the host's RDMA ports (two agents of the step, limited to RDMA, which needs the NIXL package and a usable RDMA port); with `--fabric`, copies into each GPU from another GPU's allocation imported by its fabric handle (which needs GPUs that export fabric handles), the slowest over the host's other GPUs. Each path's bandwidth is one untimed copy of `--sample-mib` MiB (default `64`) and then `--samples` timed ones (default `8`), and its per-copy cost the least time of as many copies of one page, from submission to the host's observation of the completion. A copy over a path the calibration does not state fails, naming the path; the Transfer Server measures nothing itself. A link's chunks last `transfer_chunk_us` at its bandwidth, and at least its per-copy cost, in whole pages; a copy engine or a READ keeps one chunk copying and as many more in flight as the per-copy cost lasts chunks. A link's byte budget is a share of its calibrated bandwidth, which a head halves down to 1/64 and restores by 1/8 of the bandwidth per second, so a restoration from the floor lasts at most 7.9 s (the storage protocol's `Throttle::FLOOR`, `Throttle::RESTORE_PER_SECOND` and `Throttle::restoration`).

### `load_config`

**Destination: build config.**


| Option | Default | Meaning |
| --- | --- | --- |
| `weights` | unset | A weight source (`scheme://...`) that streams the checkpoint's tensors in place of its files; the model's configuration and tokenizer stay the checkpoint's. An in-place weight update from a source sets it |

### `kv_cache_config`

**Destination: build config.**


| Option | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | Sized from storage: what `mem_fraction_static` of the device leaves once the weights, the graph pools and the process's allowance are counted | Capacity in tokens. A model that keeps a recurrent state per request keeps its checkpoints in a region of their own beside the KV cache: one checkpoint per request slot, and of the storage left beyond those and the KV cache's fixed parts, as many bytes for further checkpoints as for the KV cache (SGLang's proportion of its Mamba pool to its KV pool); with `max_tokens` stated, as many bytes as the stated capacity's pages take |
| `block_size` | `64`, the largest page the paged-attention packages read | Tokens per page of the cache group with the widest token rows, a power of two the attention kernels read; a block of the buffer pool's KV classes is one page |
| `dtype` | Each layer's compute type | `bfloat16`, `float16`, `float32` or `float8_e4m3fn`. A dense cache of another type than the layers' (a `bfloat16` model's `float16` cache) holds each key and value rounded to nearest into its type as the cache write stores it, and is read by the `triton` attention package, since trtllm-gen has kernels only for a cache of the queries' own type; that package serves those keys whatever `attention_backend` names |
| `prefix_caching` | `true` | Reuse and retain cached prompt prefixes across requests |
| `host_tier` | `true` | With `prefix_caching`, offload prompt pages to the host KV cache of each host's Storage Server and load them back when a later prompt's prefix continues past the device cache's match: every published page of a full-attention cache group, and a sliding-window group's window before each aligned boundary (`boundary_interval`) |
| `host_ratio` | `2.0` | With `host_tier`, the host KV cache's token count over the device KV cache's, the same on every rank of an epoch (as SGLang's `--hicache-ratio`): each cache group's host region holds that many tokens' pages |
| `host_gib` | None | With `host_tier`, the deployment's host KV cache on each host, in GiB, in place of `host_ratio` (as SGLang's `--hicache-size`): each cache group's host region holds as many pages as this holds over the bytes a page takes on the group's ranks. The host's Storage Server refuses a host region its available storage cannot hold |
| `mooncake` | None | With `host_tier`, keep every page of the host KV cache in Mooncake, the KV storage, and prefetch hits of at least 256 tokens back into the host KV cache at admission; the pages an epoch change evicts from the device pass through the host KV cache into Mooncake: an object with `master` (`host:port`, required), `metadata` (`"P2PHANDSHAKE"`, or an `http://`, `etcd://` or `redis://` metadata server URL), `transport` (`"tcp"` or `"rdma"`, `"tcp"`), `device` (RDMA devices, automatic when empty) and `segment_gib` (host storage each host contributes to Mooncake, `0`). It is part of the settings of the deployment's Storage Servers, whose connector opens Mooncake Store's C API (`libmooncake_store.so`, built with `WITH_STORE_C_SHARED`) from the loader's default search path and names its host by the host's own name |
| `peers` | None | With `host_tier`, the Storage Server endpoints (`host:port`, each host's `uniserve storage-server` TCP endpoint) of other hosts that serve the same checkpoint under the same served name and block size. The buffer pool prefetches a prompt's host copies from them: each peer's server holds the copies' keys and states the encoding each copy is written in, while this host's Transfer Server reads the copies of the cache's own encoding into its own host KV cache through NIXL (`docs/nixl.md`), and what did not arrive is computed. A deployment whose `dtype` is `float8_e4m3fn` also reads the pages a peer of the model's dense compute dtype (`bfloat16`) stores, into its converted-load source region of up to one request's pages (at most 1 GiB) per rank and cache group, and converts them into its FP8 pages as it loads them; a page of any other encoding is computed, and it never loads an FP8 peer's pages into a dense cache. The endpoints are not authenticated: any process that reaches a host's endpoint can read its deployments' host copies, so the hosts must share an isolated network. A list |
| `conversion_buffer_mib` | `1024` | With `peers` and an FP8 cache, the most host storage in MiB one cache group's converted-load source buffer takes on each rank: a peer's dense pages are read there before a load converts them into the cache's own encoding. It holds one request's pages of the widest row within the bound; a page it finds no slot for is computed instead |
| `boundary_interval` | `64` | With `host_tier`, the pages of the largest block size between the aligned boundaries a long prompt offloads its sliding windows before; a prompt also offloads them before the last boundary a reissue of it reuses and before its last full page. Larger intervals offload fewer windows, and a prompt that continues one offloaded earlier reuses less of it |
| `state_interval` | `256` | For a model that keeps a recurrent state per request, the decoded tokens between two checkpoints of a request's state (as SGLang's `--mamba-track-interval`); a prompt call checkpoints the last page boundary it reaches whatever the interval. A request resumed by its tokens (a migration, an epoch change, a prefix reuse) recomputes at most this many decoded tokens; a checkpoint, every recurrent layer's state, is one unit of the scheduler's recurrent-state region, a device region apart from the KV cache's |
| `state_ratio` | `1.0` | For a model that keeps a recurrent state per request, the bytes of the recurrent-state region beyond one checkpoint per request slot over the bytes of the KV planes, positive (as SGLang's `mamba_full_memory_ratio`): of what a rank's share leaves once its request slots are held, the region takes the share `ratio / (1 + ratio)` and the KV planes the rest, or, with `max_tokens` stated, `ratio` times the bytes of those tokens' pages |
| `prefetch_timeout_base` | `"1s"` | The time a copy of cached blocks the scheduler asks for may take whatever its size before it gives the copy up and the request computes those tokens, starts from noise or leaves with no stored step instead (as SGLang's `prefetch_timeout_base`): a prefetch from Mooncake or the peers, a load from or store to the host KV cache, and a stored trajectory's restore |
| `prefetch_timeout_per_ki_token` | `"250ms"` | The time such a copy may take beyond `prefetch_timeout_base` for every 1024 tokens it moves (as SGLang's `prefetch_timeout_per_ki_token`) |

### `scheduler_config`

**Destination, per field:** `policy` is a module selection (the scheduling policy); `max_queued_requests`, `chunked_prefill_size`, `max_batch` and `max_num_batched_tokens` are package parameters of that policy, bounded by the build config's limits, and the same numbers are the limit each scheduler obeys, which a change applies at each scheduler's next step within the batch the running processes were started for; `policy_budget` is what the engine's policy host runs the module within: the fuel of each call and the bounds of the limits a policy may declare.

**Read at:** `policy` selects the policy package's backend at `crates/daemon/src/epoch.rs:1311`; the other options are the policy package's arguments at `:1449-1466`, where the captured decode graphs' rows and the deployment's resident requests bound `max_batch`. `max_num_batched_tokens` is read at `:1439-1446`, and a value above the catalog is refused at load rather than clamped.


| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"arrival"` | The scheduling policy package's backend: `arrival`, the package this tree holds (`crates/scheduling/policies/arrival`), or any policy package installed at the Build Server, which refuses a backend it holds no package of |
| `order` | The package's (`"arrival"`) | The order the policy's waiting queue admits by, its `order` argument: `"arrival"` (the instant each request reached the main process) or `"priority"` (the request's own priority, lower first, then arrival) |
| `max_queued_requests` | Unbounded | Waiting requests, and finished ones with undelivered events, before a submission is refused as overloaded |
| `chunked_prefill_size` | `8192` | Prompt tokens one request prefills in one step; token models only |
| `max_batch` | `128` | Calls assembled into one forward batch, clamped to what the workers accept |
| `policy_budget.call_fuel` | `10000000` | The fuel (wasmtime's count of executed instructions) one call into the scheduling policy module may consume, and each export the engine enters for that call; a call that runs past its fuel is the module's fault, which ends its instance. Far above what a real policy's step consumes, and a bound on a call that never returns |
| `policy_budget.message_fuel_per_byte` | `200` | The fuel a call is given beyond `call_fuel` for every byte of its message, so that a message that grows with the cache (an instantiation's snapshot of every cached block) is decoded whole: typed JSON decoding in a wasm32 module costs 50 to 100 fuel a byte |
| `policy_budget.max_timeout` | `"30s"` | The longest starvation timeout a policy may declare among its limits (as `sched_ext` caps a BPF scheduler's `timeout_ms`); a policy that declares a longer one is refused |
| `policy_budget.min_prefetch_tokens` | `256` | The smallest prefix-hit deepening a policy may declare worth a prefetch, in tokens; a shorter hit is computed faster than it is fetched, and a policy that declares less is refused |

### `speculative_config`

**Destination, per field:** `num_draft_tokens` is build config (it sizes the draft the main process builds); `expected_speedup` and `refill_token_cost` are package parameters of the scheduling policy.

**Read at:** `num_draft_tokens` where the draft is built; `expected_speedup` and `refill_token_cost` become the policy package's arguments at `crates/daemon/src/epoch.rs:1469-1470`, which the policy reads as its `Parameters` (`crates/scheduling/src/limits.rs`); a parameter a deployment does not state takes the default the policy package's manifest declares (`[parameters]` of its `package.toml`).


Speculative decoding with the model's own drafter, the multi-token prediction layer a checkpoint declares (`mtp_num_hidden_layers`); token models only. Every step, each decoding request's verification evaluates its committed token followed by the draft tokens the drafter proposed after its last step and commits the longest prefix of drafts the model's own selections agree with, plus the model's token after it, so a request generates the tokens it would without speculation, one or more per step. The scheduler chooses each request's draft count every step: the drafts proposed, at most `num_draft_tokens` and one fewer than the tokens the request may still generate. The drafter keeps a cache of its own per request, the draft cache: optional state held in its own group of the KV cache, published, reused and offloaded to the host KV cache as the other groups' pages are, and released from a running request when another request's required pages need the units. A request without its draft cache (a prompt prefix reused without it, a request an epoch change moved from an epoch that did not speculate, a draft cache released for another request) verifies no drafts, a verification of one token, and decodes as without speculation; a refill then evaluates the model again over the positions before the draft cache entries the request wrote itself, in chunks of the token budget a prefill step leaves, when it is worth its cost: when the request's remaining tokens times `1 - 1 / g` exceed the refill's tokens times `k`, both sides in decode steps, with `g` the speedup speculation gives a request and `k` the time of evaluating one context token. Each scheduler measures both from what its epoch executes, as averages over about the last sixteen observations: a verification carrying drafts commits `s` tokens, the workers' decoding step takes `v` with drafts and `d` without, and a prompt step `p` per token; speculating, a request takes `v / s` per token in place of `d`, so `g = s * d / v` and `k = p / d`, and while every step has carried drafts `d` is `v`. An epoch whose verification steps cost as many decode steps as they commit tokens (`g` at most `1`) refills nothing; until it has measured what it needs, an epoch refills, which measures it. A new request writes its draft cache while it prefills and speculates from its first decoding step. A deployment that speculates serves every batch eagerly, without captured graphs. Each position of a verification samples as a decode at that position would, with the uniform its request's seed and position address, and a draft is accepted exactly when that sample equals it: the drafter proposes the argmax of its own logits, so this is the rejection rule for a deterministic drafter, and a request's outputs keep the model's distribution; a greedy request generates the tokens it would without speculation, up to the rounding of the decode kernels, which reduce a verification's rows in another order than a decode of one row (a near-even split may then fall the other way, as it may between two decode batch sizes), and a seeded sampled request reproduces. A model that keeps a recurrent state per request (Qwen3.6-35B-A3B's gated delta networks) advances it over a verification's accepted tokens only, and refills no draft cache: its request without one verifies no drafts. Speculation needs the checkpoint's multi-token-prediction layers (`mtp_num_hidden_layers` above zero), which load only when it is set, and a deployment whose experts are not split across ranks. The observation store counts `uniserve_spec_verify_rows_total`, `uniserve_spec_verify_draft_tokens_total`, `uniserve_spec_verify_accepted_tokens_total`, `uniserve_spec_verify_rejected_tokens_total` and `uniserve_spec_verify_committed_tokens_total` over the verifications that carried drafts, and states `uniserve_spec_decode_accepted_length`, the tokens such a verification committed on average over the last report ([admin API](admin.md#observation)).

| Option | Default | Meaning |
| --- | --- | --- |
| `num_draft_tokens` | `0` | The most draft tokens a verification carries per request; `0` does not speculate. An epoch that speculates keeps the draft cache in its pool and runs the drafter, so setting or changing it is an epoch change |
| `expected_speedup` | Measured | The refill policy's speedup `g`, stated in place of the measured one, at least `1`; a change of it replaces the policy package's program |
| `refill_token_cost` | Measured | The refill policy's time `k` of evaluating one context token in a refill, in decode steps, stated in place of the measured one, not negative; a change of it replaces the policy package's program |

### `migration_config`

**Destination: host layer.**


A token request that generates text moves to the engine that serves next when its epoch parks for an epoch change or a rollback, or loses the workers that hold it; it continues after the last token it delivered, with the same output and usage.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_migrations` | `3` | Migrations caused by lost workers, a lost engine process or a failed epoch on trial that a request survives before it fails; migrations of an epoch change or a rollback are not counted |

### `update_config`

**Destination: the daemon's own options.**

How an update keeps and judges what it replaces ([admin API](admin.md#updates)): how long its trial lasts before it commits by itself, how the requests it migrates continue, what the previous epoch keeps for a rollback, and when the trial rolls back without the operator. The options govern every update of the deployment; a change of them is a change of the daemon's own options, a hot restart.

| Option | Default | Meaning |
| --- | --- | --- |
| `trial` | `"30s"` | How long an update is on trial: an update that no fault and no threshold rolled back by then commits, the kept side is released, and its change converges |
| `in_flight` | `"keep"` | How the requests an epoch's switch migrates continue on the target when it computes other numerics than the epoch they ran on: `keep`, each continuing with the state it reached, which the target reads as its own (the pages it writes afterwards reach no shared tier); `recompute`, each computing its state again from its tokens, exactly |
| `previous_epoch` | `"parked"` | What the previous epoch keeps while its successor is on trial: `parked`, its drained processes, communicators, graph storage and weights, so that a rollback serves it again at once; `weights`, its weights alone, its processes exiting as the trial begins and the rest of its storage released, so that a rollback starts it again on them; `host`, as `weights` with the weights its successor does not map moved to host storage through each host's Transfer Server, and a rollback returns them to the GPUs as it starts the previous epoch again. A host whose Storage Server runs no Transfer Server, or that cannot hold them under its host-storage rule, keeps them on its GPUs, and the update's `log` events report the previous epoch kept as `weights` with each host's reason (`epoch_form`'s `previous_epoch_kept` and `host_refused`, [admin API](admin.md#status)) |
| `drain_s` | `5` | Seconds a serving engine steps its running requests at an epoch switch, beside the target that was promoted before it: a request that ends within them ends there, and one still running then ends as migratable and continues on the target. It bounds the time both epochs share each GPU; zero ends every running request at the drain. A rollback drains the target with the same deadline |
| `transfer_slowdown` | Unset | How much longer than before the change a serving scheduler's step may take while the successor prepares, as a fraction (`0.1` for 10%), past which the leader halves the byte budgets of the links the change's transfers use on every node, each second while it lasts, and restores them once every scheduler's steps are back within it. It has no default: measure it on the target hardware and workload (below); unset, the transfers take their links' whole budgets |
| `baseline` | `60` | The span of serving steps before a preparation, in seconds, that each scheduler's baseline step times for `transfer_slowdown` are read from |
| `baseline_batches` | `20` | The fewest batches of one batch-size range that make a scheduler's baseline step time in that range, so that a range the load visited for a few steps sets none |
| `rollback.failure_rate` | Unset | The share of the requests that end during the trial, in (0, 1], past which a trial whose requests failed rolls back |
| `rollback.step_time_regression` | Unset | How much longer than the previous epoch's average step before the switch the successor's may take, as a fraction (`0.2` for 20%), before the trial rolls back |
| `rollback.throughput_regression` | Unset | How much lower than the previous epoch's token throughput before the switch the successor's may fall, as a fraction, before the trial rolls back |
| `rollback.window` | Unset | The span step time and throughput are averaged over, up to the switch and up to now during the trial; a regression decides nothing before a whole span of the trial passed, or for a previous epoch that ran no batch in its span; required with either regression |

A trial also rolls back, whatever these state, on a fault of the successor: an error that leaves a main process unusable (an illegal memory access), a stalled job, non-finite logits, or a main process's exit. None of the thresholds has a default: each is measured on the deployment's own workload. A threshold that is passed reverts the update as a fault does, with the threshold's name as the failure's `kind` and what it read as the `reason`; the `trial` field of the update's `log` events states the trial ([admin API](admin.md#status)): the thresholds, the requests that ended and failed, and each replica's baseline and rates. The same thresholds decide a generation update on probation.

`transfer_slowdown` compares each scheduler's steps with its own steps of the same size before the preparation (`crates/daemon/src/interference.rs`). A step's time is a report's host-observed batch time over its batches, the step time the rollback thresholds read; a step's size is the power-of-two range of the report's query tokens per batch, since a step computing more tokens takes longer whatever the transfers do. A scheduler's baseline in a range is its batch time per batch over the `baseline` before the preparation, once its reports there hold `baseline_batches` batches; a report in a range without a baseline decides nothing. The leader decides as each report arrives, on the reports that arrived since its previous decision. The update's `log` events give the latest ratio and the halvings and restorations the leader sent (`transfers`, [admin API](admin.md#status)). The halving takes every link a task of the deployment copies over at that moment (a deployment changes one epoch at a time), and the restoration every link a halving reduced, which the Transfer Server grows back by 1/8 of its bandwidth per second; the end of the preparation restores them too.

How a value is measured, as one measurement on GB200 chose 0.1 for its workload (protocol, raw data and tables in `artifacts/transfer-interference/`): Qwen3-30B-A3B served on one GPU in BF16 under a closed-loop decode load of 64 concurrent requests of 128 prompt and 1024 output tokens (68 query tokens per step, 20.5 ms steps), while the Transfer Server copied into that GPU at a link's whole budget and at 1/4, 1/16 and 1/64 of it, 30 s per point, in three runs, each with a fresh server. The step time is the one the leader reads, per once-a-second report.

| Copies into the serving GPU | Budget 1 | 1/4 | 1/16 | 1/64 |
| --- | --- | --- | --- | --- |
| Within the GPU (1.82 TB/s at budget 1) | ×2.96 | ×1.60 | ×1.14 | ×1.03 |
| From host storage (209 GB/s) | ×1.16 | ×1.02 | ×1.00 | ×1.00 |
| From another GPU of the host (779 GB/s) | ×1.08 | ×1.01 | ×1.00 | ×1.00 |

Each entry is the median over the point's reports of the step time over its run's median without copies (the three runs agree within 0.04). Without copies, a report's step time departs from the median by at most 4.4% in 99% of reports, and one report of 354 exceeds it by more than 10% (15%). Read as the leader decides, one report at a time, a tolerance of 0.1 is exceeded by 0.3% of the reports without copies, by every report of copies within the GPU at budgets down to 1/16 and by 98% of those from host storage at its whole budget, which slow the steps by 14% or more, and by 10% of the reports of copies from another GPU at its whole budget, which slow them by 8%. A tolerance of 0.05 would be exceeded by 0.8% of the reports without copies and by up to 20% of those of copies that slow the steps by 3% or less (within the GPU at 1/64 of its budget), which 0.1 exceeds in at most 5%; one of 0.15 would let copies within the GPU at 1/16 of their budget (14%) through in 59% of the reports. Other hardware, models and loads are measured the same way before a value is stated for them.

### `device_config`

**Destination: build config** — it decides where ranks run, by the rule that places `worker_groups` and `engine_host` there.


`device`, `devices`, `worker_hosts` and `queue_depth` place ranks without a deployment layout, so each is refused together with `worker_groups`, as is `tensor_parallel_size`.

| Option | Default | Meaning |
| --- | --- | --- |
| `device` | `"cuda"` | Device kind of every rank: `cuda` numbers each host's GPUs from zero (`cuda:<index>`), `cpu` places every rank of a host on its CPU, and `meta` builds the modules' shapes without storage |
| `devices` | `[]` | Each rank's CUDA device index on its host, in rank order; empty numbers each host's ranks from zero. A change of the parallel configuration that does not state it takes the order that leaves the most weights on the GPUs already holding them, and reports it |
| `worker_hosts` | `[]` | Hosts after this one the ranks spread across, in equal blocks |
| `queue_depth` | `2` | Batches in flight per rank; a deployment group states its own |
| `transfer` | Derived per edge | Product transport per worker edge: `source[:rank]->destination[:rank]=mechanisms`, comma-separated |

### `moe_config`

**Destination, per field:** `expert_parallel` and `microbatches` are build config; `exchange` is a module selection (expert exchange).

**Read at:** `exchange` in `modules_of` (`crates/daemon/src/epoch.rs`), where it becomes the expert-exchange operator's backend name for a deployment with `expert_parallel`; `moe_runner_backend` beside it.


[Expert parallelism](expert_parallelism.md) describes each exchange.

| Option | Default | Meaning |
| --- | --- | --- |
| `expert_parallel` | `false` | Shard routed experts across the deployment's expert workers, across schedulers of one rank each (`data_parallel_size` above one), or otherwise over the tensor-parallel ranks of each scheduler's worker group, each rank holding whole experts; a group split over pipeline stages or sequence members is refused with it |
| `exchange` | unset | The expert-exchange package's backend: `alltoall`, `megamoe` or `deepep` among the packages this tree holds (`crates/ops/kernels/expert_exchange`), refused at the Build Server when it holds no package of the backend; unset, a deployment whose schedulers exchange experts runs `alltoall`. A deployment that shards its groups' experts over their tensor-parallel ranks moves no token and refuses an exchange |
| `microbatches` | `1` | Microbatches overlapping attention with dedicated expert workers |
| `expert_order` | The checkpoint's order | The experts' order (`[moe_config.expert_order]`): for each routed expert layer, by its module path, the logical expert IDs in the order of the physical slots that hold them, across the layer's expert ranks; a layer it does not name keeps the checkpoint's order. A change takes the `order` effect: it is prepared on the copy stream beside the batches and committed at one computation boundary, the epoch, the graphs' addresses and the requests staying ([admin API](admin.md#expert-redistribution)). An epoch starts with the order stated |
| `beside_rows` | `1024` | The rows per rank of the middle capture class of an expert union's steps, a step in which another member prefills a few prompts while this one decodes, beside the class of its decode buckets and the class of the union's whole capacity; not captured when a decode bucket reaches it or it lies within a quarter of the whole capacity |

### `cuda_graph_config`

**Destination: build config.**


| Option | Default | Meaning |
| --- | --- | --- |
| `decode_batch_sizes` | Worker catalog | Decode batch sizes captured, strictly increasing; the largest decode batch the worker admits is captured besides them |
| `prefill_token_sizes` | Worker catalog | Prefill token counts captured, strictly increasing; the batch token bound is captured besides them, for every row bucket up to the largest batch the worker admits |

### `worker_config`

**Destination: build config.**


| Option | Default | Meaning |
| --- | --- | --- |
| `engine` | `uniserve` | The engine package the deployment's main processes run as: the name of an implementation of the engine interface installed at the Build Server (`crates/engine/package.toml` is `uniserve`). The daemon fixes its revision when the configuration is applied, has the Build Server of every host the deployment runs on build the executable for that host's platform before any process starts, and each node's daemon resolves that revision at its own Build Server and starts it as `uniserve-engine --launch <path>`. A change of it changes the epoch's engine, so the change is applied by an epoch switch ([Epochs](architecture.md#epochs)) — the target executable starts as the next generation of the same group and rank, parks, is promoted in place, and serves once the previous epoch has drained — and the epoch's build identity is the one its first child reports |
| `engine_revision` | unset | The revision of the engine package to run. Unset takes the latest installed revision when the deployment starts and keeps the serving one afterwards: a revision installed later runs only once a configuration names it here. A change of it is an epoch switch |
| `watchdog_timeout` | `"150s"` | Time a job may stay in flight on a worker before the worker reports it stalled, which fails the epoch: the worker group is replaced, and an update on trial rolls back |
| `join_timeout` | `"5m"` | How long the main processes of an epoch whose ranks span several hosts wait for one another as they start: the scheduler node for every peer's registration, a peer for the scheduler node's acceptance, and every process for the others at the ranks' host-group rendezvous. A process that connected and then exits or loses its host fails the start at once; the bound covers a process that never arrives |
| `compare_timeout` | `"5m"` | Time a generation update's shadow pass waits for the running requests of every scheduler to run a step its changed kernels are compared on, and for every peer main process to end its own preparation of the generation; an update not compared within it is abandoned |
| `mesh` | Unset | `tower=gen:<device>` places the generation tower on another device than the worker's |
| `report_period` | `"1s"` | How often each main process states its metrics and each scheduler its statistics to the daemon, each report covering the period since the last |
| `copy_budget` | `"512MiB"` | The bytes, summed over a worker group's ranks, that the copies started with one batch may move together on the devices' copy streams beside its forward; a larger copy starts alone. The default bounds the copies to 2.6 ms of one GB200 GPU's 209 GB/s link to host storage, an eighth of a 20.5 ms decode step, measured with the interference measurement under `update_config.transfer_slowdown`; a host of another link bandwidth or step time states its own |
| `host_threads` | Half the CPUs the process may run on | The threads of a main process's host work: the width of its media pool and the threads a conversion of a request's condition media shares its work over. Unset, half the CPUs its affinity mask holds within its cgroup's CPU quota, at least one, the other half left to its workers', schedulers' and communicators' threads |
| `host_queue_depth` | `4` | The tasks of host work one thread of the media pool has room for: a host phase (media read, encodes, muxing) serves `host_threads` times this many requests at once |
| `standby` | `false` | Keep a parked standby beside every node's main process of the serving epoch. The standby is a second child of the same group, rank and epoch at the next launch generation, with its own CUDA contexts: it maps the epoch's weights, captures its graphs on the warmup scratch units, keeps its graph pools, scratch and communicators resident or suspended per `standby_storage`, and parks; it reports `Parked` when it is ready, and `Promoted` when the promotion took effect: an update's target is parked by the switch that starts it and answers those reports to it, while the standby a serving epoch keeps parks while it starts and is promoted in place when its failure domain fails, without an update in flight. When the serving main process fails with a process-fatal fault, its whole failure domain — the node's main process plus every cross-node group it is in — promotes one parked child per node, every promotion issued before any answer is awaited, and each node then starts a new standby at the generation after the promoted one. A member of the domain without a parked standby makes the domain relaunch instead, and a GPU that needs a reset takes its node's standby with it, so only migration to another node or replica helps. No standby exists while the epoch is in precopy or on probation: a fault there reverts the update, and the epoch is started again rather than promoted. Each standby child holds a CUDA context, graph pools and workspaces of its own, which the device storage `mem_fraction_static` leaves must hold. The workers a main process still starts keep their standby by the same rule until the worker process is retired: a group that is its own failure domain and runs on one node keeps a standby rank group, and the groups of an expert epoch or on several nodes keep a standby epoch of the domain |
| `standby_storage` | `auto` | How a parked standby (a node's standby child, a group's standby rank group or a failure domain's standby epoch) keeps the device storage of its graph pools, operator scratch, expert all-to-all workspaces and NCCL communicators: `resident`; `suspended`, released on the device and restored at the promotion, before its graphs run (the graph storage is copied to host storage and back with its contents at unchanged addresses; the `alltoall` exchange's MNNVL workspace is unmapped and mapped again at its addresses by every rank of the epoch together; NCCL communicators are suspended and resumed, and every engine captures their collectives without NCCL graph buffer registration (the engine process states `NCCL_GRAPH_REGISTER=0` for its NCCL library itself), since a registered buffer's peers keep mapping the storage the suspension gave back; the `deepep` exchange's windows and the colocated `megamoe` exchange's NVSHMEM heap, which cannot keep their addresses, are freed, and the promotion creates them again and captures the graphs that run expert steps anew); or `auto`, resident while every device keeps beside it a tenth of its storage, the share a serving worker leaves for graphs and activations, and suspended otherwise. The ranks of an epoch, group or domain take one mode: one rank's crowded device suspends them all. A standby that lacks that share even suspended is not kept |

A worker compiles no kernel. A main process resolves the modules its forwards launch at its host's Build Server, which builds a module its store lacks within `build_config`'s `max_jobs` and `host_reserve` ([Host services](#host-services), [The Build Server](architecture.md#the-build-server)).

### `profiling_config`

**Destination, per field:** `fence_modules`, `graph_pc_sampling` and `nvtx_ranges` are build config; `fence_probes` and `fence_probes_off` are profiling switches rather than deployment content and keep switching in place at a step boundary, as `session` starts and ends its session there.


The workers' always-on profiling: the fence probes at the entry and exit of the model's modules, exported as per-step durations by module class (`uniserve:worker_fence_step_seconds`, [execution profiling](profiling.md#fence-probes)). A profiling session is the `session` table, applied with the package as every change is ([execution profiling](profiling.md#profiling-sessions)).

| Option | Default | Meaning |
| --- | --- | --- |
| `fence_probes` | `true` | Whether the probes record; a change switches every worker's probes in place, at the engine's next step boundary |
| `graph_pc_sampling` | `false` | Whether every worker enables CUPTI's PC sampling, stopped, before it captures its CUDA graphs and keeps it enabled for its life, so that `pc_sampling` sessions attribute the samples of the graphs' kernels. CUPTI cannot run PM sampling, the range profiler or whole-graph activity records beside it, so sessions of `pm_sampling`, `range_metrics` and timelines that trace graphs whole are then refused. Each worker holds CUPTI's sampling working storage and buffer as process host storage: about 120 MiB of working storage (measured on GB200) and the smallest buffer while no session samples, and the buffer a session states (`pc_sampling.hardware_buffer_mib`, 512 MiB by default) while one samples (`docs/profiling.md`). The enabled sampler slows serving even while no session samples: on GB200 serving Qwen3-30B-A3B to 1 and to 32 concurrent requests, the step time averaged 9.6% and 11.6% more than without the option (eight points of a minute each way, from two servers each; single points from 1% to 21% more), and the step time with the option varied more between points. The graphs are captured at an epoch's start, so a change takes a new epoch |
| `fence_modules` | The model family's step stack: its decoder or denoiser stack, one instance per step, within the always-on budget `docs/profiling.md` states (Qwen3's `models.qwen3.transformer.Transformer`, MiniMax-H3's `Transformer`) | Name patterns (`*` matches any run) of the module classes whose instances carry probes, a class being the name a module gives its profiler range (`docs/profiling.md`); the probes are captured into an epoch's CUDA graphs, so a change takes a new epoch. Empty for no probe |
| `nvtx_ranges` | `false` | Whether every engine emits its NVTX ranges (the `uniserve.worker` domain, the scheduler's and the control thread's) from its epoch's start, for a profiler that records them, such as Nsight Systems; a session's `timeline` collector turns them on while it records either way. A change takes a new epoch |
| `session` | Unset | The profiling session the serving epoch's workers run (`[profiling_config.session]`: its collectors, its window and its output directory, [execution profiling](profiling.md#profiling-sessions)). While it is stated, the session starts at the engine's next step boundary and records until its window ends, its guard ends it, or its epoch stops executing; a change ends the running session, every worker writing its file, and starts the new one, and removing it ends the running one. Each state the session enters is a `log` event of source `profile` ([admin API](admin.md#status)) |
| `fence_probes_off` | `[]` | Name patterns (`*` matches any run) of the probed classes whose probes stay off while `fence_probes` is on, for classes whose probes cost more than the always-on budget; a change switches every worker's probes in place, at the engine's next step boundary, as `fence_probes` does |

### `logging_config`

**Destination: host layer, daemon.**


| Option | Default | Meaning |
| --- | --- | --- |
| `log_requests` | `false` | Log a summary line for each completed request |
| `log_stats` | `true` | Log engine statistics periodically: request throughput, scheduler occupancy, cache usage and prefix-cache hit rate, as structured fields |
| `stats_interval` | `"10s"` | Interval of the statistics log line, over which its throughput is averaged |

### `chat_config`

**Destination: host layer.**


| Option | Default | Meaning |
| --- | --- | --- |
| `template` | The checkpoint's | The chat template, read exactly as written: `{ file = "/path/chat.jinja" }` names a file of Jinja source, `{ inline = "..." }` states the source itself. Unset, the checkpoint's own template renders: its standalone template file, else the `chat_template` of its `tokenizer_config.json` |
| `template_kwargs` | Unset | Keyword arguments bound into every chat-template render |
| `template_content_format` | `"auto"` | How message content renders: `string` or `openai`; `auto` detects it from the template, keeping each message's own shape for a template that reads both |
| `reasoning_parser` | `"auto"` | `auto` splits `reasoning` from `content`; `none` streams reasoning delimiters as content |

### `image_fetch_config`

**Destination: host layer.**


Input images of chat requests. The table takes the options of [`media_config`](#media_config), the one schema of request media, with these defaults where the two differ: `timeout` `"20s"`, `max_bytes` `20000000` (the per-image limit OpenAI's API states), and no `max_request_bytes`, so a request's images are bounded each and not in total.

A chat family that takes images (Qwen3.6-35B-A3B, BF16 and NVFP4) serves `image_url` parts, URLs and `data:` URIs alike, through the vision tower its configuration declares (`vision_config`): every deployment serves a declared tower, and a checkpoint that declares one without its weights or its processor files (`preprocessor_config.json`, `video_preprocessor_config.json`) is refused at load. Each image is sized by the processor's rule (`smart_resize` to whole 32-pixel merged tokens within the processor's pixel bounds, its EXIF orientation applied) and takes as many cache tokens as merged tokens, which count against `max_model_len` and, with the prompt, against `max_num_batched_tokens`. The rank holds eight images' features at once, each slot sized for the largest image the processor admits, so a request reads at most eight images. Every other family refuses an image part with `unsupported feature image_input`.

### `video_config`

**Destination: build config.**


The [FastH3 guide](minimax_h3/minimax_h3.md#startup-and-capacity) explains how these options size startup and capacity.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_seconds` | The model family's longest output (15 for MiniMax-H3) | Longest admitted output, in seconds: a positive number, which the served model also holds to its own bounds when its request processor binds (4 to 15 for MiniMax-H3) |
| `sizes` | The model family's sizes (`["1344x768", "768x1344"]` for MiniMax-H3) | Output sizes startup prepares and requests may name, each `WxH` in pixels, the first the size of a request that names none; the served model refuses at startup a size its checkpoint does not generate |
| `text_capacities` | `1024`, then steps of 2048, below `max_model_len` | Prompt capacities startup captures, strictly increasing and none above `max_model_len`; `max_model_len` itself completes them as the largest, so every admitted prompt runs in the smallest capacity that holds it |
| `max_condition_rows` | Largest condition set | Denoiser rows one request's conditions may take |
| `latent_page_bytes` | `1048576` (one MiB) | Bytes of one page of a denoising rank's host latent region, where a preempted request's trajectory is saved; a whole number of float32 samples. A saved trajectory takes whole pages |

### `media_config`

**Destination: host layer.**


Condition media of video requests. `image_fetch_config` takes the same options with its own defaults.

| Option | Default | Meaning |
| --- | --- | --- |
| `directory` | Unset | Directory `file://` media resolves under, stated by its canonical path (an absolute path through no symbolic link), which `file://` URLs name it by; `file://` is refused when unset |
| `allow_remote` | `true` | Fetch `http(s)://` media |
| `allow_private` | `false` | Allow `http(s)` URLs resolving to an address outside IANA's global unicast space or in a block of its special-purpose registries: loopback, private, shared, link-local (cloud metadata services), unique-local, documentation and the rest; IPv6 forms that carry an IPv4 address are judged by it |
| `timeout` | `"60s"` | Time to fetch one `http(s)` medium, redirects and body included |
| `connect_timeout` | `"10s"` | Time to establish one connection |
| `max_redirects` | `3` | Redirects one fetch follows; each hop is checked again, and none may go from `https` to `http` |
| `max_concurrent_fetches` | `64` | `http(s)` downloads in progress at once; further fetches wait |
| `max_bytes` | `"256MiB"` | Largest medium, fetched, read or `data:` |
| `max_request_bytes` | `"256MiB"` | Media one request may carry in total; the video routes' body limit follows it |
| `probe_timeout` | `"30s"` | Time a demuxer may take to read one audio or video medium's streams |
| `max_concurrent_probes` | The daemon's available parallelism | Media probed at once across requests |

Reference videos are decoded through the FFmpeg libraries the engine executable links, with no external program.

### `video_job_config`

**Destination: host layer.**


| Option | Default | Meaning |
| --- | --- | --- |
| `retention` | `"1h"` | Time a finished asynchronous job stays retrievable |
| `max_bytes` | `"1GiB"` | Generated video the asynchronous jobs keep in shared memory |

## Deployment layout

The `worker_groups` option is the deployment layout: it places worker groups on devices and hosts and names the blocks each runs. It is part of the configuration tree, one `[[worker_groups]]` table per group and one `[worker_groups.blocks.<name>]` table per block of the group; the packages of `configs/minimax_h3` state a layout each beside the model they serve, as `configs/minimax_h3/fasth3-ulysses4x2/config.toml` does. A change states the layout it moves to in its own package.

```toml
# FastH3 on eight GPUs of one host: two Ulysses4 denoiser replicas, each with
# its own TP4 text encoder.

[[worker_groups]]
id = "text"
replicas = 2
ranks = ["cuda:0-7"]
queue_depth = 6
mem_fraction_static = 0.26

[worker_groups.blocks.text_encoder]
tensor_parallel_size = 4

[[worker_groups]]
id = "flow"
replicas = 2
ranks = ["cuda:0-7"]
queue_depth = 6
mem_fraction_static = 0.73

[worker_groups.blocks.transformer]
sequence_parallel = { kind = "ulysses", ulysses_degree = 4, projection = "exchange" }

[worker_groups.blocks.video_decoder]
distribution = "temporal_units"

[worker_groups.blocks.audio_decoder]
distribution = "temporal_units"

[worker_groups.blocks.latent_encoder]
distribution = "temporal_units"

# The host media blocks, on each replica's CUDA ranks.
[worker_groups.blocks.video_codec]
distribution = "temporal_units"

[worker_groups.blocks.muxer]
ranks = [0]

[worker_groups.blocks.media_reader]
ranks = [0]
```

The host media blocks, the H.264 encoders (`video_codec`), the muxer and the media reader, run on the host inside the main process of the ranks they are placed on, and a main process serves CUDA devices only, so a deployment places them on CUDA ranks: a rank on `cpu` is refused when the node is launched.

A `[[worker_groups]]` table takes:

| Key | Default | Meaning |
| --- | --- | --- |
| `id` | Required | Group name; replica `i` of a replicated group is `<id>-<i>` |
| `role` | `model` | `model` for a group serving the model's blocks; `prefill` and `decode` for a prefill group and a decode group of the text model (below); `experts` for dedicated expert ranks, which take no blocks |
| `replicas` | `1` | Copies of the group, each over the next equal block of `ranks`. The copies' blocks are replicas of the same blocks under one scheduler, which binds each request to one copy |
| `ranks` | Required | The group's ranks, every replica's together, in rank order |
| `queue_depth` | Required | Batches in flight per rank. A rank that keeps video requests resident takes three positions per request, so such a group holds up to `queue_depth / 3` requests at once; a group serving several replicas, such as a shared text encoder, needs positions for all of them |
| `mem_fraction_static` | Top-level `mem_fraction_static` | Each rank's share of its GPU's storage for this group, which bounds what the rank divides. The deployment's quota of a GPU is limited to the shares of the ranks placed there: each rank of a group that states its own adds it, the ranks of groups that state none hold the top-level fraction together, and the sum is at most the whole GPU |

A `prefill` group prefills prompts only and a `decode` group decodes only; a deployment configures both or neither, and each places the `model` block alone. A request prefills on a prefill group; once its prompt is complete and its first token sampled, the host's Storage Server copies the request's cache units, device to device, from the prefill group's KV cache into the same units of a decode group's (a transfer task of the Transfer Server, reading a prefill group of another host through NIXL), and the request decodes there from its first token. One scheduler drives both groups as separate workers, each group's ranks running only that group's batches, so the two groups' devices compute side by side. The groups must have the same number of ranks, which startup checks, and the same KV cache layout, which the ranks agree at startup: every rank of the deployment sizes its cache to the same units. A group on another host (`gb200-b/cuda:0`) needs a deployment of several nodes, whose Storage Servers listen on TCP, with NIXL on both hosts' library search path ([NIXL transfers](nixl.md)). Such a deployment generates text: canvas requests are refused.

```toml
[[worker_groups]]
id = "prefill"
role = "prefill"
ranks = ["cuda:0"]
queue_depth = 2

[worker_groups.blocks.model]

[[worker_groups]]
id = "decode"
role = "decode"
ranks = ["cuda:1"]
queue_depth = 2

[worker_groups.blocks.model]
```

A `ranks` entry is `[HOST/]DEVICE`. `DEVICE` is one device (`cuda:0`, `cpu`), an inclusive index range (`cuda:0-3`) or a device repeated for CPU workers (`cpu*4`). Without `HOST` the ranks run on the leader's host, the first of `hosts`; `gb200-b/cuda:0-3` places four ranks on the node of `hosts` whose host name is `gb200-b` ([Nodes](#nodes)).

A block table takes:

| Key | Default | Meaning |
| --- | --- | --- |
| `ranks` | Every rank of the replica | Indices into one replica's ranks; their order assigns each rank its position in the parallel degrees or the unit schedule. A denoiser block's positions run over the Ulysses degree fastest, then the tensor degree, then the all-gather degree, and the pipeline degree slowest: in a block of `tensor_parallel_size = 2` and Ulysses degree 2, positions 0 and 1 are one Ulysses group and positions 0 and 2 one tensor group |
| `tensor_parallel_size`, `pipeline_parallel_size` | `1` | Model-parallel degrees |
| `sequence_parallel` | `{ kind = "local" }` | `{ kind = "ulysses", ulysses_degree = N }` with optional `projection` and `projection_rows`, `{ kind = "allgather", allgather_degree = N }`, or `{ kind = "hybrid", ulysses_degree = N, allgather_degree = M }` |
| `distribution` | Unset | `temporal_units` runs its own copy of the block on every rank and deals independent media units among them. Each block admits the forms its owner declares, which startup checks: the model family's for a part of its pipeline (MiniMax-H3's video decoder deals one unit per rank, its text encoder and denoisers take none), the engine's for the host blocks `media_reader` and `muxer` (one rank, no distribution) and `video_codec` (one unit per rank) |
| `units_per_rank` | `1` | Consecutive units each rank takes under `distribution` |
| `weights` | `device` | `host` keeps the dense projection weights of the block's repeated layer stack in page-locked host storage and serves it without CUDA graphs |

Without a distribution, a block's ranks number the product of its degrees. `configs/` holds the published deployments, and the [FastH3 guide](minimax_h3/minimax_h3.md#deployments) lists the FastH3 files.

A text model's `model` block splits the model along one axis. `tensor_parallel_size` splits every layer's heads and projections. `pipeline_parallel_size` gives each rank, in the block's rank order, an equal contiguous share of the decoder layers, each share holding as many attention layers as the others, and every rank keeps the embedding, the head, the drafter and the vision tower whole; the stages of a step run one after another, the last stage's final rows reaching every stage, and a group's stages lie on one host. A `ulysses` sequence split gives each rank a share of every batch's rows and of the attention and key/value heads, which its degree must divide, under either projection; it serves neither a model whose layers keep a recurrent state nor speculative decoding, and `projection_rows` is refused. A block with two degrees above one, the `allgather` and `hybrid` kinds, and serving groups of one deployment that split the model differently are refused at startup.
