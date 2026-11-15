# Configuration

A UniServe server is configured by one tree of serving options and, for a parallel configuration other than one tensor-parallel worker group per scheduler, a deployment file. The tree's top level holds the options most deployments state; the rest sit in `*_config` tables, one per part of the system they configure.

## Command line

`uniserve serve` reads options from three places, each overriding the last:

1. **`--config FILE`**: a TOML file of the tree.
2. **Flags** for the most common options, each a shortcut for one path of the tree.
3. **`--set PATH=VALUE`**, repeatable, reaching any option by its path.

Options none of them states keep their defaults. `GET /admin/config` returns the resolved non-default options as reloadable TOML. Reading it restores defaults without treating options belonging to other model families or rank-assignment modes as explicitly configured.

```bash
uniserve serve "$H3_MODEL" --worker-groups configs/minimax_h3/ulysses4.toml \
  --max-running-requests 2 --max-model-len 10240 \
  --set 'video_config.text_capacities=[1024]'

uniserve serve --config fasth3.toml --set video_config.max_seconds=10
```

| Flag | Path |
| --- | --- |
| `MODEL` (positional) | `model` |
| `--host`, `--port` | `admin_config.host`, `admin_config.port` |
| `--served-model-name` | `served_model_name` |
| `--worker-groups` | `worker_groups` |
| `--tensor-parallel-size`, `--data-parallel-size` | `tensor_parallel_size`, `data_parallel_size` |
| `--max-running-requests` | `max_running_requests` |
| `--max-model-len` | `max_model_len` |
| `--lora-path` | `lora_path` |
| `--mem-fraction-static` | `mem_fraction_static` |
| `--host-identity` | `host_identity` |
| `--nnodes`, `--node-rank`, `--dist-init-addr` | `nnodes`, `node_rank`, `dist_init_addr` |

`--ready-fd FD` names an inherited descriptor the daemon writes its readiness record to: one JSON line once its listeners are bound and one once it serves, each stating the endpoints it bound (`docs/admin.md`, Listeners). It is no option of the configuration.

`--log-level` sets the log level of every target of the `uniserve` process, and `--log-level-http` the level of the daemon's HTTP stack (the server and the media-fetching client); every subcommand takes them, before or after its name. A level is one of tracing's names (`off`, `error`, `warn`, `info`, `debug`, `trace`), `info` when unset, and any other value is refused. The environment sets no level (`RUST_LOG` is not read). The daemon starts its engine processes at its own `--log-level`, and a Storage Server its helpers at its own; lines name the level as tracing does (`WARN`, not `WARNING`).

A `--set` value is a TOML value (`2`, `true`, `"text"`, `[1024, 4096]`, `{ mode = "quality" }`) or JSON, and otherwise the text itself, so `--set quantization.mode=quality` and `--set admin_config.host=0.0.0.0` need no quotes. A list is a TOML array in a `--set` value as in a file (`--set 'video_config.sizes=["1344x768", "768x1344"]'`): quote the assignment, since the shell may expand brackets. One invocation may set a path several times, the last winning, but not both a table and an option inside it.

Every option is checked by its type wherever it is stated, and an error names its path: an unknown option, a value its option does not take, or zero for a positive count. A duration is whole seconds or a value with units (`500ms`, `30s`, `5m`, `1h`). A byte size is whole bytes or an integer with a decimal (`20MB`) or binary (`512MiB`) unit. Paths in options resolve against the server's working directory.

Each host's Storage Server (`uniserve storage-server serve`, see `architecture.md`) is a host service: one server per host and user, on `storage-server.sock` in UniServe's directory of the user's runtime directory (`$XDG_RUNTIME_DIR/uniserve`), where the running server also publishes its readiness record (`storage-server.ready`). It serves with the settings of one document (`storage::Settings`, TOML), which a deployment states (`storage_config` below, with the connector of `kv_cache_config.mooncake`): the host's daemon starts the server with them when the host has none (`storage-server serve --daemon --settings -`, the document on its standard input), and refuses to share a server that serves with other settings, naming both, rather than serving with whichever settings its first starter gave; such a server is stopped (`uniserve storage-server stop`) or its settings stated. An operator starts one by hand with `uniserve storage-server serve [--settings FILE] [--ready-fd FD] [--daemon]`: the document's `[server]` table holds `storage_config`'s keys and its `[connector]` table `kv_cache_config.mooncake`'s, every key at its default when unstated, and a key no table knows is refused. The server binds its socket and its TCP endpoint before anything else (port `0` allowed) and reports both in its readiness record (`storage`, `peer`) with its settings: on `--ready-fd`, or, with `--daemon`, on its starter's standard output once it serves. `uniserve storage-server status` prints that record from the running server, and `uniserve storage-server stop` stops it and returns once it ended. `uniserve storage-server list` prints what it holds, with each deployment's buffer pool on lines led by `buffer pool of NAME:` (the blocks of each region and class that are shared, private or retired, each worker group's last completed batch and the stamps its deferred releases, commits, transfers and copy ends wait for, and the requests whose copies are in flight), and `uniserve storage-server retire --deployment NAME` releases the storage of a deployment that is no longer served. A deployment is named by its served model name, so a restarted server finds the storage it left under the same name. A deployment's host KV cache is the host regions of its buffer pool, one memfd per rank index, which the server admits against the host's storage when a session first states the region. When a deployment claims a host, the server retires every other deployment whose daemon is no longer connected, so a host keeps the weights and caches of the deployments it serves only.

Each host's Build Server (`uniserve build-server serve`, see `architecture.md`) is configured by `build_config`, whose keys are the flags of its `serve` command: `socket` (the user's `build-server.sock` unless stated), `package_root` and `store_roots` (`build-server/packages` and `build-server/modules` in `$XDG_DATA_HOME/uniserve`, else `~/.local/share/uniserve`; the first store root is written, the rest are read-only shared roots), `upstream` (the socket of the upstream Build Server that accepts `install`, on a host that is not it), `read_only_stores` (store directories this server serves and never writes, each as `build --store` makes one with its `packages` and `modules`: the store a release ships with the modules of its own packages, whose revisions and modules a host serves with no toolchain, so a first start builds nothing; a revision the host installs is numbered after them), `targets` (the GPU architectures it builds for, the host's devices' unless stated; a target is one the Build Server's target description `crates/build-server/python/src/uniserve_build_server/targets.toml` states), `max_jobs` (the builds that run at once, and the compiler processes each build runs, 4 unless stated), `host_reserve` (the host memory in bytes a build leaves free, 8 GiB unless stated) and `peak_memory` (a table of toolchain kind to the bytes one build of it reserves: `triton` 8 GiB, `cute_dsl` 10 GiB, `nvcc` 4 GiB and `cargo` the larger of `max_jobs` × 2 GiB and 16 GiB unless stated; `--peak-memory KIND=BYTES` on the command line). No option names a toolchain: the server runs its Python environment's program `uniserve-build-server toolchain --json` from the search path once when it starts (`crates/build-server/python/README.md`), which describes the Triton and CuTe DSL compilers the environment locks, the CUDA toolkit of the release its lock fixes (found through the `cudart-<release>` pkg-config module NVIDIA's packages register) and the host C++ compiler; a host without the environment refuses those packages with the reason. cargo is rustup's, at the release the workspace's `rust-toolchain.toml` pins. The server keeps what builds share in `build-server` of the user's cache directory (`$XDG_CACHE_HOME/uniserve`, else `~/.cache/uniserve`): the pinned third-party sources, cargo's target directory, the vendored registry sources of each lockfile and the builds' workspaces. The daemon prepares an epoch's modules at `build_config.socket`, and starts the host's Build Server with these settings when none serves there; a Build Server that serves already keeps the settings it was started with. `uniserve build-server install DIR` installs a package on the upstream Build Server and `uniserve build-server list` prints the installed packages, revisions and modules, and `uniserve build-server resolve OP BACKEND REVISION` prints where the store holds one module's files, building the module first when it is missing. `uniserve build-server build --keys FILE [--keys FILE ...] --packages DIR --target SM --store DIR [--max-jobs N] [--peak-memory KIND=BYTES]` builds a store with a Build Server of its own, in process, which holds the store's package root while it runs: for every `operator<TAB>descriptor key` line of the key lists it installs the operator's packages from the package tree `--packages` (one directory per operator, one package directory per backend) into `DIR/packages`, builds the key for the target into `DIR/modules` with at most `--max-jobs` builds at once (4 unless stated), prints `ok` or `FAILED` per key, and fails with the keys that did not build. A package whose build inputs are unchanged since its latest revision in the store keeps that revision, so building more keys into an existing store extends it; a change of the package's files, of the toolchain's version, or of the Build Server environment's compile driver, pins or compilers (`uniserve_build_server`) installs a new revision whose modules are built again. Every module is stored under the digest of the inputs it was built from (`<op>/<backend>/<revision>/<inputs>/<target>/<key>`), so a module is never served for other inputs, whichever package root numbered its revision; every installed revision records its inputs, and a package root holding a revision without its record is refused. `DIR/packages` and `DIR/modules` are then a `serve`'s `--package-root` and `--store-root`.

`uniserve keys MODEL [OPTIONS]` prints the key list a deployment's main processes launch, in the format `build-server build --keys` reads (`operator<TAB>descriptor key` lines, sorted, each key once), and exits, so a store can be built for a deployment before it serves. It takes `serve`'s options, resolves them as `serve` does, and runs each main process's own start on this node: the same family, rank placement, capacity, model and runners, up to the profile run every device's runners make at start, in which each runner runs the forwards it captures and runs eagerly under a meta context that reports each launch's descriptor key instead of launching it (`architecture.md`, The profile run). The model is loaded from its checkpoint's safetensors headers, so no weight is read; no daemon, Storage Server or NCCL communicator is started. Each main process is the executable of the engine package the deployment selects, run with `--keys`: the command fixes that package's revision at this host's Build Server, which it starts when none serves and which builds the executable when its store misses it. The deployment's devices are used, because the ranks' resident memory and the runners' buffers are device allocations, so they must be free. Logs go to standard error and the key list to standard output:

```bash
uniserve keys /workspace/models/Qwen3-30B-A3B --max-model-len 2048 --max-running-requests 8 \
  --set dtype=bfloat16 --set attention_backend=trtllm > keys.txt \
  && uniserve build-server build --keys keys.txt --packages crates/ops/kernels --target sm_100a \
       --store /var/lib/uniserve/store
```

`build --package DIR` installs a package that fills a whole interface with one module, the engine (`crates/engine`) or a scheduling policy, and builds that module for its own target (the host's Rust target triple, or `wasm32-unknown-unknown`) with cargo; it is repeatable and combines with `--keys`.

Some options configure one kind of model, by what the served model family serves: `max_num_batched_tokens`, `scheduler_config.chunked_prefill_size`, `speculative_config`, `chat_config` and `image_fetch_config` configure chat models (Qwen3-MoE); `video_config`, `media_config` and `video_job_config` configure video generation (MiniMax-H3). Startup refuses such an option stated for a model of the other kind, naming the option and the served family.

## Configuration file

A configuration file is the tree in TOML, as `GET /admin/config` returns it:

```toml
model = "/workspace/models/FastVideo-FastH3-8-Step-V2"
served_model_name = "FastH3"
worker_groups = "configs/minimax_h3/ulysses4.toml"
max_running_requests = 2

[server_config]
host = "0.0.0.0"

[quantization]
mode = "quality"

[video_config]
text_capacities = [1024, 10240]

[video_job_config]
retention = "30m"
```

## Options

Each option's destination follows what its value is used for: a value that builds the main process is **build config**; one that selects an implementation is a **module selection**; one that only steers a decision is a **package parameter**; what the daemon reads stays in the **host layer**. An option a section does not name separately has the destination stated under that section's heading; the sections whose options split state it per field below.

Three backend-name fields are read with `attention_backend` (`modules_of`, `crates/daemon/src/epoch.rs`): `prefill_attention_backend` and `decode_attention_backend` each override it for their phase, and `moe_runner_backend` selects the grouped expert GEMM. A backend name selects the package of that name at the Build Server for every descriptor key it covers; a key it does not cover, and every key of an operator whose backend is unset or `auto`, takes the installed package that covers it with the highest manifest `priority` (`architecture.md`, the Build Server). A named backend no installed package has is refused when the epoch's modules are prepared.

### Top level

| Destination | Options |
| --- | --- |
| Deployment identity, cannot change | `model`, `served_model_name` |
| Build config | `max_model_len`, `lora_path`, `dtype`, `quantization`, `max_running_requests`, `mem_fraction_static`, `tensor_parallel_size`, `data_parallel_size`, `worker_groups`, `engine_host` |
| Host layer, daemon (a node's startup arguments) | `host_identity`, `nnodes`, `node_rank`, `dist_init_addr` |
| Host layer | `mm_processor_kwargs` |
| Module selection | `attention_backend` (attention), with `prefill_attention_backend` and `decode_attention_backend` overriding it per phase |
| Package parameter | `max_num_batched_tokens` (the scheduling policy's bound); read at `crates/daemon/src/epoch.rs:1364-1366` |
| Retired, refused by name | `quantization_config`: the option is stated as `quantization`, and the retired spelling is refused with the name to state instead |

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | Required | Local model directory or Hugging Face repository |
| `served_model_name` | `model` | The single model name the API serves |
| `max_model_len` | The checkpoint's `max_position_embeddings`, 8192 when it declares none; a media pipeline (MiniMax-H3) has none and its deployment must state it | Longest sequence a request may reach, in tokens; for a media pipeline the longest prompt presentation it admits |
| `lora_path` | Unset | A weight adapter fused into the checkpoint's weights at load and serving every request: an adapter file, or a directory holding `adapter_model.safetensors`; video families only, and the family must recognize it ([FastH3 adapters](minimax_h3/minimax_h3.md#fasth3-adapters)) |
| `dtype` | `"bfloat16"` | Element type of parameters and activations: `bfloat16`, `float16` or `float32` |
| `quantization` | `{}` | Precision policy: `mode` (`quality`, `balanced`, `performance`, `maximum`) and representations selected by module under `modules` (`bf16`, `fp16`, `fp8`, `nvfp4`); empty selects the checkpoint's own quantization. The model family interprets it when the engine opens the checkpoint and refuses a policy it does not serve, naming each module: MiniMax-H3 serves every mode and selection but `video_vae = "bf16"` over a dense checkpoint, converting its FP8 and NVFP4 modules' projections at load and measuring each activation's scale on every call ([FastH3 precision](minimax_h3/minimax_h3.md#precision)); the chat families state no mode and no module, and a checkpoint that declares its own quantization takes no policy |
| `mm_processor_kwargs` | `{}` | Options of the model's multimodal processor; MiniMax-H3 takes `reference_image_short_edge` and `enlarge_references` ([FastH3 guide](minimax_h3/minimax_h3.md#startup-and-capacity)) |
| `attention_backend` | `"auto"` | Attention package: `auto` has the Build Server select, for each attention descriptor key the deployment launches, the installed package that covers it by manifest priority; a package's backend name (`trtllm`, `triton`) serves every key it covers, and the others are selected as for `auto` |
| `max_running_requests` | `128` | Concurrently resident requests, clamped to what the workers hold: each worker group holds this many request slots, or as many as its ranks' shares of their GPUs hold beside the weights, the graph pools and the fewest KV units when fewer fit (a slot holds a token request's slot state and, for a model that keeps a recurrent state, one checkpoint of it, or a video request's trajectory and its products on the rank), and a start whose share holds no slot fails, naming the bytes one slot needs |
| `max_num_batched_tokens` | The widest prefill batch the workers' captured graphs hold; `8192` when they capture none | Token work admitted in one scheduling step, the scheduling policy's step budget when its package states none; token models only |
| `mem_fraction_static` | `0.7` | The deployment's share of each device's storage for weights, static allocations, graph pools and the KV cache, in (0, 1]: the limit `⌊total × fraction⌋` of the GPU's quota, which both epochs of a configuration change and the host's Transfer Server share (`docs/admin.md`); a deployment group's `mem_fraction_static` overrides it for the group's ranks, and a GPU that holds ranks of several groups has the shares of them all (below) |
| `tensor_parallel_size` | `1` | Tensor-parallel workers of each scheduler's group without `worker_groups` |
| `data_parallel_size` | `1` | DP ranks, each a scheduler with its own KV cache and worker groups. Independent replicas each run an engine of their own, and the leader sends a request to the one with the fewest in flight. With `moe_config.expert_parallel`, replicas of one rank each (`tensor_parallel_size` 1) share their experts and exchange tokens, so one engine runs every replica's scheduler and the scheduling policy's `route` names each request's; replicas of several tensor-parallel ranks each shard their experts over their own ranks and stay independent replicas. With `worker_groups`, the file's groups that serve requests form the schedulers as equal consecutive blocks |
| `worker_groups` | Unset | [Deployment file](#deployment-files) placing worker groups and their blocks |
| `host_identity` | `"localhost"` | This node's host name; its daemon runs the ranks assigned to this host |
| `engine_host` | This node's identity | The joined node that runs the engine and any ranks without an explicit host. Changing it starts a successor epoch there while the HTTP daemon remains on its node |
| `nnodes` | `1` | Nodes of the deployment, each running one `uniserve serve` |
| `node_rank` | `0` | This node's rank. The leader, rank 0, serves the API and requires `model`; the other node daemons join it and launch engines or workers assigned to their hosts |
| `dist_init_addr` | Unset | The leader's join endpoint (`HOST:PORT`): on the leader, the host the other nodes reach it at and the port it binds on every interface when it starts (`0` picks one, which its readiness record states as `join`); on another node, the `join` endpoint of the leader's record. Required with several nodes |

A checkpoint that declares its own quantization (ModelOpt's `quantization_config` in `config.json` or `hf_quant_config.json`) is read in the representation each weight's declaration and shard state. NVFP4 (`quant_algo: NVFP4`) runs NVFP4 products over activations encoded to NVFP4. A `MIXED_PRECISION` export states one algorithm per module in `quantized_layers`, and each module is read in its own: `FP8` is a per-tensor E4M3 product whose activation is encoded at the calibrated static `input_scale`; `NVFP4` as above; `W4A16_NVFP4` keeps activations in 16 bits, and its weight is read as the BF16 matrix its codes and scales state (`code · block_scale · tensor_scale`, rounded once), so its products are 16-bit products. That weight is resident in BF16: a `W4A16_NVFP4` export does not keep its 4-bit memory footprint, and the products differ from a 16-bit-activation FP4 kernel's, which applies the tensor scale to its accumulator, by at most half a BF16 unit per weight. Another per-module algorithm is refused by name. A packed ModelOpt export of a diffusers pipeline declares its quantization in `modelopt_manifest.json` at its root instead: its calibrated projections are stored as `weight_packed`, `weight_scale` and `weight_tensor_scale`, run the same NVFP4 products, and encode their activations at the tensor scale the manifest states for each module (`activation_amax / (6 · 448)`). The key/value cache keeps the element type `kv_cache_config` states, the activations' otherwise.

### `server_config`

**Destination: host layer.** The leader daemon's application server reads these: how the HTTP and OpenAI API answer; where the daemon listens is `admin_config`'s.


| Option | Default | Meaning |
| --- | --- | --- |
| `api_key` | Unset | Bearer token every route but `/health` and `/metrics` requires; `UNISERVE_API_KEY` supplies it when unset or blank. Never printed |
| `request_timeout` | Unset | Time until a response head is sent, answered with 504 when it expires; streamed bodies are not bounded |
| `max_concurrent_requests` | Unset | In-flight chat and synchronous video requests above which requests are shed with 503 |
| `shutdown_timeout` | `"30s"` | Longest drain of in-flight requests at shutdown; `0` stops at once |
| `request_id_headers` | `false` | Add an `X-Request-Id` response header with the ID the request ran under |

### `admin_config`

**Destination: the daemon's own options.** The daemon's listeners read it: the client API's address, and the two Unix sockets the admin API is served on apart from it ([admin API](admin.md#listeners)). `uniserve serve --host` and `--port` set the address. The daemon binds every listener when it starts and states the endpoints it bound in its readiness record.

| Option | Default | Meaning |
| --- | --- | --- |
| `host` | `"127.0.0.1"` | TCP address the daemon's client API listens on |
| `port` | `0` | TCP port the daemon's client API listens on; `0` picks a free one, which the readiness record states (`client`) |
| `uds` | Unset | Unix domain socket the client API listens on instead of TCP; a stale socket file is replaced and the socket file is removed at shutdown |
| `admin_uds` | `admin.sock` in the daemon's instance directory (`daemon.XXXXXX` in `$XDG_RUNTIME_DIR/uniserve`) | Unix socket of the operator's admin API, created with mode `0600`; it serves every admin route |
| `harness_uds` | `harness.sock` in the daemon's instance directory | Unix socket of the harness's subset of the admin API (apply, commit, rollback, status, events and the update's report), created with mode `0660`: the operator grants a harness access through the socket's group |
| `harness_options` | `[]` | The option paths a harness may set through `PUT /admin/config` on the harness listener, beside the package selections it may always set (`worker_config.engine`, `worker_config.engine_revision`, `scheduler_config.policy`, `attention_backend`, `prefill_attention_backend`, `decode_attention_backend`, `moe_runner_backend`, `moe_config.exchange`) |

### `storage_config`

**Destination: the host's Storage Server.** The settings of the Storage Server of every host the deployment serves on (`[server]` of the server's settings document): a daemon that starts a host's server starts it with them, and one that finds a server serving with other settings refuses to share it. A deployment that other nodes may join (`dist_init_addr`) has its servers accept other hosts on `0.0.0.0:0` unless `tcp` states an endpoint.

| Option | Default | Meaning |
| --- | --- | --- |
| `tcp` | Unset | The TCP endpoint (`HOST:PORT`, port `0` picks one) on which the server also accepts other hosts' heads and peers, which its readiness record states as `peer`; local connections only when unset. The endpoint is not authenticated |
| `host_reserve_gib` | `10` | Host storage, in GiB, that new host storage (a buffer pool's host region, the host copy of an epoch's weights) leaves to the host's other processes: storage the host's available storage less the reserve cannot hold is refused |
| `transfer_chunk_us` | `1000` | How long one chunk of the Transfer Server's copies lasts at the link's bandwidth, in microseconds |
| `transfer_gpus` | `true` | Whether the Transfer Server copies on the host's GPUs: it then requires the CUDA driver and copies on every GPU the driver shows; `false` copies host storage alone without loading the driver |
| `child_restart_s` | `5` | Seconds the server waits before it starts a child process again (its Mooncake connector or its Transfer Server), doubled for each process in a row that ended before it connected |
| `child_restart_max_s` | `300` | The longest wait, in seconds, before a child process starts again |
| `transfer_deadline_s` | `60` | How long a buffer pool's transfer of a request between worker groups may take, in seconds, before it is cancelled and answered as failed |
| `put_deadline_s` | `60` | How long a buffer pool's put of host copies to the remote store may take, in seconds, before it is stopped and answered with the keys put so far |
| `buffer_pool_trace` | Unset | A directory the server records every buffer pool's inputs and completions in (`specs/buffer-pool.md` §18); a failure to write stops the trace, not the pool |

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
| `block_size` | Largest supported up to 64 | Tokens per page of the cache group with the widest token rows, a power of two |
| `dtype` | Each layer's compute type | `bfloat16`, `float16`, `float32` or `float8_e4m3fn`. A dense cache of another type than the layers' (a `bfloat16` model's `float16` cache) holds each key and value rounded to nearest into its type as the cache write stores it, and is read by the `triton` attention package, since trtllm-gen has kernels only for a cache of the queries' own type; that package serves those keys whatever `attention_backend` names |
| `prefix_caching` | `true` | Reuse and retain cached prompt prefixes across requests |
| `host_tier` | `true` | With `prefix_caching`, offload prompt pages to the host KV cache of each host's Storage Server and load them back when a later prompt's prefix continues past the device cache's match: every published page of a full-attention cache group, and a sliding-window group's window before each aligned boundary (`boundary_interval`) |
| `host_ratio` | `2.0` | With `host_tier`, the host KV cache's token count over the device KV cache's, the same on every rank of an epoch (as SGLang's `--hicache-ratio`): each cache group's host region holds that many tokens' pages |
| `host_gib` | None | With `host_tier`, the deployment's host KV cache on each host, in GiB, in place of `host_ratio` (as SGLang's `--hicache-size`): each cache group's host region holds as many pages as this holds over the bytes a page takes on the group's ranks. The host's Storage Server refuses a host region its available storage cannot hold |
| `mooncake` | None | With `host_tier`, keep every page of the host KV cache in Mooncake, the KV storage, and prefetch hits of at least 256 tokens back into the host KV cache at admission; the pages an epoch change evicts from the device pass through the host KV cache into Mooncake: an object with `master` (`host:port`, required), `metadata` (`"P2PHANDSHAKE"`, or an `http://`, `etcd://` or `redis://` metadata server URL), `transport` (`"tcp"` or `"rdma"`, `"tcp"`), `device` (RDMA devices, automatic when empty) and `segment_gib` (host storage each host contributes to Mooncake, `0`). It is part of the settings of the deployment's Storage Servers, whose connector opens Mooncake Store's C API (`libmooncake_store.so`, built with `WITH_STORE_C_SHARED`) from the loader's default search path and names its host by the host's own name |
| `peers` | None | With `host_tier`, the Storage Server endpoints (`host:port`, each host's `uniserve storage-server` TCP endpoint) of other hosts that serve the same checkpoint under the same served name and block size. The buffer pool prefetches a prompt's host copies from them: each peer's server holds the copies' keys and states the encoding each copy is written in, while this host's Transfer Server reads the copies of the cache's own encoding into its own host KV cache through NIXL (`docs/nixl.md`), and what did not arrive is computed. A deployment whose `dtype` is `float8_e4m3fn` also reads the pages a peer of the model's dense compute dtype (`bfloat16`) stores, into its converted-load source region of up to one request's pages (at most 1 GiB) per rank and cache group, and converts them into its FP8 pages as it loads them; a page of any other encoding is computed, and it never loads an FP8 peer's pages into a dense cache. The endpoints are not authenticated: any process that reaches a host's endpoint can read its deployments' host copies, so the hosts must share an isolated network. A list |
| `boundary_interval` | `64` | With `host_tier`, the pages of the largest block size between the aligned boundaries a long prompt offloads its sliding windows before; a prompt also offloads them before the last boundary a reissue of it reuses and before its last full page. Larger intervals offload fewer windows, and a prompt that continues one offloaded earlier reuses less of it |
| `state_interval` | `256` | For a model that keeps a recurrent state per request, the decoded tokens between two checkpoints of a request's state (as SGLang's `--mamba-track-interval`); a prompt call checkpoints the last page boundary it reaches whatever the interval. A request resumed by its tokens (a migration, an epoch change, a prefix reuse) recomputes at most this many decoded tokens; a checkpoint, every recurrent layer's state, is one unit of the scheduler's recurrent-state region, a device region apart from the KV cache's |

### `scheduler_config`

**Destination, per field:** `policy` is a module selection (the scheduling policy); `max_queued_requests`, `chunked_prefill_size`, `max_batch` and `max_num_batched_tokens` are package parameters of that policy, bounded by the build config's limits, and the same numbers are the limit each scheduler obeys, which `PUT /admin/config` applies in place within the batch the running processes were started for.

**Read at:** `policy` selects the policy package's backend at `crates/daemon/src/epoch.rs:1311`; the other options are the policy package's arguments at `:1449-1466`, where the captured decode graphs' rows and the deployment's resident requests bound `max_batch`. `max_num_batched_tokens` is read at `:1439-1446`, and a value above the catalog is refused at load rather than clamped.


| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"arrival"` | The scheduling policy package's backend: `arrival`, the package this tree holds (`crates/scheduling/policies/arrival`), or any policy package installed at the Build Server, which refuses a backend it holds no package of |
| `order` | The package's (`"arrival"`) | The order the policy's waiting queue admits by, its `order` argument: `"arrival"` (the instant each request reached the main process) or `"priority"` (the request's own priority, lower first, then arrival) |
| `max_queued_requests` | Unbounded | Waiting requests, and finished ones with undelivered events, before a submission is refused as overloaded |
| `chunked_prefill_size` | `8192` | Prompt tokens one request prefills in one step; token models only |
| `max_batch` | `128` | Calls assembled into one forward batch, clamped to what the workers accept |

### `speculative_config`

**Destination, per field:** `num_draft_tokens` is build config (it sizes the draft the main process builds); `expected_speedup` and `refill_token_cost` are package parameters of the scheduling policy.

**Read at:** `num_draft_tokens` where the draft is built; `expected_speedup` and `refill_token_cost` become the policy package's arguments at `crates/daemon/src/epoch.rs:1469-1470`, which the policy reads as its `Parameters` (`crates/scheduling/src/limits.rs`); a parameter a deployment does not state takes the default the policy package's manifest declares (`[parameters]` of its `package.toml`).


Speculative decoding with the model's own drafter, the multi-token prediction layer a checkpoint declares (`mtp_num_hidden_layers`); token models only. Every step, each decoding request's verification evaluates its committed token followed by the draft tokens the drafter proposed after its last step and commits the longest prefix of drafts the model's own selections agree with, plus the model's token after it, so a request generates the tokens it would without speculation, one or more per step. The scheduler chooses each request's draft count every step: the drafts proposed, at most `num_draft_tokens` and one fewer than the tokens the request may still generate. The drafter keeps a cache of its own per request, the draft cache: optional state held in its own group of the KV cache, published, reused and offloaded to the host KV cache as the other groups' pages are, and released from a running request when another request's required pages need the units. A request without its draft cache (a prompt prefix reused without it, a request an epoch change moved from an epoch that did not speculate, a draft cache released for another request) verifies no drafts, a verification of one token, and decodes as without speculation; a refill then evaluates the model again over the positions before the draft cache entries the request wrote itself, in chunks of the token budget a prefill step leaves, when it is worth its cost: when the request's remaining tokens times `1 - 1 / g` exceed the refill's tokens times `k`, both sides in decode steps, with `g` the speedup speculation gives a request and `k` the time of evaluating one context token. Each scheduler measures both from what its epoch executes, as averages over about the last sixteen observations: a verification carrying drafts commits `s` tokens, the workers' decoding step takes `v` with drafts and `d` without, and a prompt step `p` per token; speculating, a request takes `v / s` per token in place of `d`, so `g = s * d / v` and `k = p / d`, and while every step has carried drafts `d` is `v`. An epoch whose verification steps cost as many decode steps as they commit tokens (`g` at most `1`) refills nothing; until it has measured what it needs, an epoch refills, which measures it. A new request writes its draft cache while it prefills and speculates from its first decoding step. A deployment that speculates serves every batch eagerly, without captured graphs. Each position of a verification samples as a decode at that position would, with the uniform its request's seed and position address, and a draft is accepted exactly when that sample equals it: the drafter proposes the argmax of its own logits, so this is the rejection rule for a deterministic drafter, and a request's outputs keep the model's distribution; a greedy request generates the tokens it would without speculation, up to the rounding of the decode kernels, which reduce a verification's rows in another order than a decode of one row (a near-even split may then fall the other way, as it may between two decode batch sizes), and a seeded sampled request reproduces. A model that keeps a recurrent state per request (Qwen3.6-35B-A3B's gated delta networks) advances it over a verification's accepted tokens only, and refills no draft cache: its request without one verifies no drafts. Speculation needs the checkpoint's multi-token-prediction layers (`mtp_num_hidden_layers` above zero), which load only when it is set, and a deployment whose experts are not split across ranks. `/metrics` reports `uniserve:worker_spec_verify_rows`, `_draft_tokens`, `_accepted_tokens`, `_rejected_tokens` and `_committed_tokens` over the verifications that carried drafts, and `uniserve:spec_decode_accepted_length`, the tokens such a verification committed on average over the last report.

| Option | Default | Meaning |
| --- | --- | --- |
| `num_draft_tokens` | `0` | The most draft tokens a verification carries per request; `0` does not speculate. An epoch that speculates keeps the draft cache in its pool and runs the drafter, so setting or changing it is an epoch change (`PUT /admin/config`) |
| `expected_speedup` | Measured | The refill policy's speedup `g`, stated in place of the measured one, at least `1`; a change of it moves the policy package at `PUT /admin/config` |
| `refill_token_cost` | Measured | The refill policy's time `k` of evaluating one context token in a refill, in decode steps, stated in place of the measured one, not negative; a change of it moves the policy package at `PUT /admin/config` |

### `migration_config`

**Destination: host layer.**


A token request that generates text moves to the engine that serves next when its epoch parks for an epoch change or a rollback, or loses the workers that hold it; it continues after the last token it delivered, with the same output and usage.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_migrations` | `3` | Migrations caused by lost workers, a lost engine process or a failed epoch on trial that a request survives before it fails; migrations of an epoch change or a rollback are not counted |

### `update_config`

**Destination: the daemon's own options.**

How an update keeps and judges what it replaces (`PUT /admin/config`, [admin API](admin.md#updates)). An option that an update's own `set` states governs that update alone; the configuration keeps its own. An update's deadline is stated with the apply itself (`deadline_s`), not here.

| Option | Default | Meaning |
| --- | --- | --- |
| `previous_epoch` | `"parked"` | What the previous epoch keeps while its successor is on trial: `parked`, its drained processes, communicators, graph storage and weights, so that a rollback serves it again at once; `weights`, its weights alone, its processes exiting as the trial begins and the rest of its storage released, so that a rollback starts it again on them; `host`, as `weights` with the weights its successor does not map moved to host storage through each host's Transfer Server, and a rollback returns them to the GPUs as it starts the previous epoch again. A host whose Storage Server runs no Transfer Server, or that cannot hold them under its host-storage rule, keeps them on its GPUs, and the update's status reports the previous epoch kept as `weights` with each host's reason (`previous_epoch_kept`, `host_refused`) |
| `drain_s` | `5` | Seconds a serving engine steps its running requests at an epoch switch, beside the target that was promoted before it: a request that ends within them ends there, and one still running then ends as migratable and continues on the target. It bounds the time both epochs share each GPU; zero ends every running request at the drain. A rollback drains the target with the same deadline |
| `transfer_slowdown` | `0.1` | How much longer than before the change a serving scheduler's step may take while the successor prepares, as a fraction (`0.1` for 10%), past which the leader halves the byte budgets of the links the change's transfers use on every node, each second while it lasts, and restores them once every scheduler's steps are back within it; the default is measured on GB200 (below) |
| `rollback.failure_rate` | Unset | The share of the requests that end during the trial, in (0, 1], past which a trial whose requests failed rolls back |
| `rollback.step_time_regression` | Unset | How much longer than the previous epoch's average step before the switch the successor's may take, as a fraction (`0.2` for 20%), before the trial rolls back |
| `rollback.throughput_regression` | Unset | How much lower than the previous epoch's token throughput before the switch the successor's may fall, as a fraction, before the trial rolls back |
| `rollback.window` | Unset | The span step time and throughput are averaged over, up to the switch and up to now during the trial; a regression decides nothing before a whole span of the trial passed, or for a previous epoch that ran no batch in its span; required with either regression |

A trial also rolls back, whatever these state, on a fault of the successor: an error that leaves a main process unusable (an illegal memory access), a stalled job, non-finite logits, or a main process's exit. None of the thresholds has a default: each is measured on the deployment's own workload. A threshold that is passed reverts the update as a fault does, with the threshold's name as the failure's `kind` and what it read as the `reason`; `GET /admin/config` reports the trial under `update.trial`: the thresholds, the requests that ended and failed, and each replica's baseline and rates. The same thresholds decide a generation update on probation.

`transfer_slowdown` compares each scheduler's steps with its own steps of the same size before the preparation (`crates/daemon/src/interference.rs`). A step's time is a once-a-second report's host-observed batch time over its batches, the step time the rollback thresholds read; a step's size is the power-of-two range of the report's query tokens per batch, since a step computing more tokens takes longer whatever the transfers do. A scheduler's baseline in a range is its batch time per batch over the minute before the preparation, once its reports there hold 20 batches; a report in a range without a baseline decides nothing. The update's status gives the latest ratio and the halvings and restorations the leader sent (`transfers`). The halving takes every link a task of the deployment copies over at that moment (a deployment changes one epoch at a time), and the restoration every link a halving reduced, which the Transfer Server grows back by 1/8 of its bandwidth per second; the end of the preparation restores them too.

The default, 0.1, comes from a measurement on GB200 (`crates/storage-server/examples/transfer_interference.rs`; protocol, raw data and tables in `artifacts/transfer-interference/`): Qwen3-30B-A3B served on one GPU in BF16 under a closed-loop decode load of 64 concurrent requests of 128 prompt and 1024 output tokens (68 query tokens per step, 20.5 ms steps), while the Transfer Server copied into that GPU at a link's whole budget and at 1/4, 1/16 and 1/64 of it, 30 s per point, in three runs, each with a fresh server. The step time is the one the leader reads, per once-a-second report.

| Copies into the serving GPU | Budget 1 | 1/4 | 1/16 | 1/64 |
| --- | --- | --- | --- | --- |
| Within the GPU (1.82 TB/s at budget 1) | ×2.96 | ×1.60 | ×1.14 | ×1.03 |
| From host storage (209 GB/s) | ×1.16 | ×1.02 | ×1.00 | ×1.00 |
| From another GPU of the host (779 GB/s) | ×1.08 | ×1.01 | ×1.00 | ×1.00 |

Each entry is the median over the point's reports of the step time over its run's median without copies (the three runs agree within 0.04). Without copies, a report's step time departs from the median by at most 4.4% in 99% of reports, and one report of 354 exceeds it by more than 10% (15%). Read as the leader decides, one report at a time, a tolerance of 0.1 is exceeded by 0.3% of the reports without copies, by every report of copies within the GPU at budgets down to 1/16 and by 98% of those from host storage at its whole budget, which slow the steps by 14% or more, and by 10% of the reports of copies from another GPU at its whole budget, which slow them by 8%. A tolerance of 0.05 would be exceeded by 0.8% of the reports without copies and by up to 20% of those of copies that slow the steps by 3% or less (within the GPU at 1/64 of its budget), which 0.1 exceeds in at most 5%; one of 0.15 would let copies within the GPU at 1/16 of their budget (14%) through in 59% of the reports. Other hardware, models and loads are measured the same way before the value is changed.

### `device_config`

**Destination: build config** — it decides where ranks run, by the rule that places `worker_groups` and `engine_host` there.


`device`, `devices`, `worker_hosts` and `queue_depth` place ranks without a deployment file, so each is refused together with `worker_groups`, as is `tensor_parallel_size`.

| Option | Default | Meaning |
| --- | --- | --- |
| `device` | `"cuda"` | Device kind of every rank: `cuda` numbers each host's GPUs from zero (`cuda:<index>`), `cpu` places every rank of a host on its CPU, and `meta` builds the modules' shapes without storage |
| `devices` | `[]` | Each rank's CUDA device index on its host, in rank order; empty numbers each host's ranks from zero. A change of the parallel configuration (`PUT /admin/config`) that does not state it takes the order that leaves the most weights on the GPUs already holding them, and reports it |
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
| `exchange` | `"alltoall"` | The expert-exchange package's backend: `alltoall`, `megamoe` or `deepep` among the packages this tree holds (`crates/ops/kernels/expert_exchange`), refused at the Build Server when it holds no package of the backend; a deployment that shards its groups' experts over their tensor-parallel ranks moves no token and refuses any but the default |
| `microbatches` | `1` | Microbatches overlapping attention with dedicated expert workers |

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
| `watchdog_timeout` | `"5m"` | Time a batch may run without progress of its workers before the worker group is replaced; a worker reports a stalled job after half of it |
| `mesh` | Unset | `tower=gen:<device>` places the generation tower on another device than the worker's |
| `standby` | `false` | Keep a parked standby beside every node's main process of the serving epoch. The standby is a second child of the same group, rank and epoch at the next launch generation, with its own CUDA contexts: it maps the epoch's weights, captures its graphs on the warmup scratch units, keeps its graph pools, scratch and communicators resident or suspended per `standby_storage`, and parks; it reports `Parked` when it is ready, and `Promoted` when the promotion took effect: an update's target is parked by the switch that starts it and answers those reports to it, while the standby a serving epoch keeps parks while it starts and is promoted in place when its failure domain fails, without an update in flight. When the serving main process fails with a process-fatal fault, its whole failure domain — the node's main process plus every cross-node group it is in — promotes one parked child per node, every promotion issued before any answer is awaited, and each node then starts a new standby at the generation after the promoted one. A member of the domain without a parked standby makes the domain relaunch instead, and a GPU that needs a reset takes its node's standby with it, so only migration to another node or replica helps. No standby exists while the epoch is in precopy or on probation: a fault there reverts the update, and the epoch is started again rather than promoted. Each standby child holds a CUDA context, graph pools and workspaces of its own, which the device storage `mem_fraction_static` leaves must hold. The workers a main process still starts keep their standby by the same rule until the worker process is retired: a group that is its own failure domain and runs on one node keeps a standby rank group, and the groups of an expert epoch or on several nodes keep a standby epoch of the domain |
| `standby_storage` | `auto` | How a parked standby (a node's standby child, a group's standby rank group or a failure domain's standby epoch) keeps the device storage of its graph pools, operator scratch, expert all-to-all workspaces and NCCL communicators: `resident`; `suspended`, released on the device and restored at the promotion, before its graphs run (the graph storage is copied to host storage and back with its contents at unchanged addresses; the `alltoall` exchange's MNNVL workspace is unmapped and mapped again at its addresses by every rank of the epoch together; NCCL communicators are suspended and resumed, and every engine captures their collectives without NCCL graph buffer registration (the engine process states `NCCL_GRAPH_REGISTER=0` for its NCCL library itself), since a registered buffer's peers keep mapping the storage the suspension gave back; the `deepep` exchange's windows and the colocated `megamoe` exchange's NVSHMEM heap, which cannot keep their addresses, are freed, and the promotion creates them again and captures the graphs that run expert steps anew); or `auto`, resident while every device keeps beside it a tenth of its storage, the share a serving worker leaves for graphs and activations, and suspended otherwise. The ranks of an epoch, group or domain take one mode: one rank's crowded device suspends them all. A standby that lacks that share even suspended is not kept |

A worker compiles no kernel. A main process resolves the modules its forwards launch at its host's Build Server, which builds a module its store lacks within `build_config`'s `max_jobs` and `host_reserve` ([Command line](#command-line), [The Build Server](architecture.md#the-build-server)).

### `profiling_config`

**Destination, per field:** `fence_modules` and `graph_pc_sampling` are build config; `fence_probes` and `fence_probes_off` are profiling switches rather than deployment content and keep switching in place at a step boundary.


The workers' always-on profiling: the fence probes at the entry and exit of the model's modules, exported as per-step durations by module class (`uniserve:worker_fence_step_seconds`, [execution profiling](profiling.md#fence-probes)). Profiling sessions take their options with their request (`POST /admin/profile`).

| Option | Default | Meaning |
| --- | --- | --- |
| `fence_probes` | `true` | Whether the probes record; a change switches every worker's probes in place, at the engine's next step boundary |
| `graph_pc_sampling` | `false` | Whether every worker enables CUPTI's PC sampling, stopped, before it captures its CUDA graphs and keeps it enabled for its life, so that `pc_sampling` sessions attribute the samples of the graphs' kernels. CUPTI cannot run PM sampling, the range profiler or whole-graph activity records beside it, so sessions of `pm_sampling`, `range_metrics` and timelines that trace graphs whole are then refused. Each worker holds CUPTI's sampling buffer and working storage as process host storage: about 129 MiB at the default period (131,584 KiB measured on GB200), and up to 512 MiB more while a session samples at a short period (`docs/profiling.md`). The enabled sampler slows serving even while no session samples: on GB200 serving Qwen3-30B-A3B to 1 and to 32 concurrent requests, the step time averaged 9.6% and 11.6% more than without the option (eight points of a minute each way, from two servers each; single points from 1% to 21% more), and the step time with the option varied more between points. The graphs are captured at an epoch's start, so a change takes a new epoch |
| `fence_modules` | The model family's step stack: its decoder or denoiser stack, one instance per step, within the always-on budget `docs/profiling.md` states (Qwen3's `models.qwen3.transformer.Transformer`, MiniMax-H3's `Transformer`) | Name patterns (`*` matches any run) of the module classes whose instances carry probes, a class being the name a module gives its profiler range (`docs/profiling.md`); the probes are captured into an epoch's CUDA graphs, so a change takes a new epoch. Empty for no probe |
| `fence_probes_off` | `[]` | Name patterns (`*` matches any run) of the probed classes whose probes stay off while `fence_probes` is on, for classes whose probes cost more than the always-on budget; a change switches every worker's probes in place, at the engine's next step boundary, as `fence_probes` does |

### `logging_config`

**Destination: host layer, daemon.**


| Option | Default | Meaning |
| --- | --- | --- |
| `log_requests` | `false` | Log a summary line for each completed request |
| `log_stats` | `true` | Log engine statistics periodically |

### `chat_config`

**Destination: host layer.**


| Option | Default | Meaning |
| --- | --- | --- |
| `template` | The checkpoint's | Chat template, as a file path or the template itself |
| `template_kwargs` | Unset | Keyword arguments bound into every chat-template render |
| `template_content_format` | `"auto"` | How message content renders: `auto`, `string` or `openai` |
| `reasoning_parser` | `"auto"` | `auto` splits `reasoning_content` from `content`; `none` streams reasoning delimiters as content |

### `image_fetch_config`

**Destination: host layer.**


| Option | Default | Meaning |
| --- | --- | --- |
| `timeout` | `"20s"` | Time to fetch one `http(s)` image URL, redirects and body included |
| `max_bytes` | `20000000` | Largest input image, fetched or `data:` |
| `allow_private` | `false` | Allow image URLs resolving to loopback, private, link-local, unique-local or cloud metadata addresses |

A chat family that takes images (Qwen3.6-35B-A3B, BF16 and NVFP4) serves `image_url` parts, URLs and `data:` URIs alike, when its checkpoint holds the vision tower and its processor files (`preprocessor_config.json`, `video_preprocessor_config.json`). Each image is sized by the processor's rule (`smart_resize` to whole 32-pixel merged tokens within the processor's pixel bounds, its EXIF orientation applied) and takes as many cache tokens as merged tokens, which count against `max_model_len` and, with the prompt, against `max_num_batched_tokens`. The rank holds eight images' features at once, each slot sized for the largest image the processor admits, so a request reads at most eight images. Every other family refuses an image part with `unsupported feature image_input`.

### `video_config`

**Destination: build config.**


The [FastH3 guide](minimax_h3/minimax_h3.md#startup-and-capacity) explains how these options size startup and capacity.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_seconds` | The model family's longest output (15 for MiniMax-H3) | Longest admitted output, in seconds: a positive number, which the served model also holds to its own bounds when its request processor binds (4 to 15 for MiniMax-H3) |
| `sizes` | The model family's sizes (`["1344x768", "768x1344"]` for MiniMax-H3) | Output sizes startup prepares and requests may name, each `WxH` in pixels; the served model refuses at startup a size its checkpoint does not generate |
| `text_capacities` | `1024`, then steps of 2048, below `max_model_len` | Prompt capacities startup captures, strictly increasing and none above `max_model_len`; `max_model_len` itself completes them as the largest, so every admitted prompt runs in the smallest capacity that holds it |
| `max_condition_rows` | Largest condition set | Denoiser rows one request's conditions may take |

### `media_config`

**Destination: host layer.**


| Option | Default | Meaning |
| --- | --- | --- |
| `directory` | Unset | Directory `file://` condition media resolves under; `file://` is refused when unset |
| `allow_remote` | `true` | Fetch `http(s)://` condition media |
| `max_request_bytes` | `"256MiB"` | Condition media one request may carry in total; the video routes' body limit follows it |

Reference videos are decoded through the FFmpeg libraries the engine executable links, with no external program.

### `video_job_config`

**Destination: host layer.**


| Option | Default | Meaning |
| --- | --- | --- |
| `retention` | `"1h"` | Time a finished asynchronous job stays retrievable |
| `max_bytes` | `"1GiB"` | Generated video the asynchronous jobs keep in shared memory |

## Deployment files

A configuration update reads the deployment file again, so editing its contents at the same path can move ranks to other devices or hosts. The resolved configuration carries those contents to the engine; subsequent file edits do not change the ranks that configuration launches.

A deployment file, the `worker_groups` option, places worker groups on devices and hosts and names the blocks each runs. It is TOML: one `[[worker_groups]]` table per group and one `[worker_groups.blocks.<name>]` table per block of the group, as in `configs/minimax_h3/fasth3-ulysses4x2.toml`:

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
sequence_parallel = { kind = "ulysses", ulysses_degree = 4 }

[worker_groups.blocks.video_decoder]

[worker_groups.blocks.audio_decoder]
distribution = "temporal_units"

[worker_groups.blocks.latent_encoder]
distribution = "temporal_units"

[[worker_groups]]
id = "encoder"
replicas = 2
ranks = ["cpu*8"]
queue_depth = 6

[worker_groups.blocks.video_codec]

[[worker_groups]]
id = "reader"
ranks = ["cpu"]
queue_depth = 6

[worker_groups.blocks.media_reader]

[[worker_groups]]
id = "muxer"
ranks = ["cpu"]
queue_depth = 12

[worker_groups.blocks.muxer]
```

A `[[worker_groups]]` table takes:

| Key | Default | Meaning |
| --- | --- | --- |
| `id` | Required | Group name; replica `i` of a replicated group is `<id>-<i>` |
| `role` | `model` | `model` for a group serving the model's blocks; `prefill` and `decode` for a prefill group and a decode group of the text model (below); `experts` for dedicated expert ranks, which take no blocks |
| `replicas` | `1` | Copies of the group, each over the next equal block of `ranks`. The copies' blocks are replicas of the same blocks under one scheduler, which binds each request to one copy |
| `ranks` | Required | The group's ranks, every replica's together, in rank order |
| `queue_depth` | Required | Batches in flight per rank. A rank that keeps video requests resident takes three positions per request, so such a group holds up to `queue_depth / 3` requests at once; a group serving several replicas, such as a shared text encoder, needs positions for all of them |
| `mem_fraction_static` | Top-level `mem_fraction_static` | Each rank's share of its GPU's storage for this group, which bounds what the rank divides. The deployment's quota of a GPU is limited to the shares of the ranks placed there: each rank of a group that states its own adds it, the ranks of groups that state none hold the top-level fraction together, and the sum is at most the whole GPU |

A `prefill` group prefills prompts only and a `decode` group decodes only; a deployment configures both or neither, and each places the `model` block alone. A request prefills on a prefill group; once its prompt is complete and its first token sampled, the host's Storage Server copies the request's cache units, device to device, from the prefill group's KV cache into the same units of a decode group's (a transfer task of the Transfer Server, reading a prefill group of another host through NIXL), and the request decodes there from its first token. One scheduler drives both groups as separate workers, each group's ranks running only that group's batches, so the two groups' devices compute side by side. The groups must have the same number of ranks, which startup checks, and the same KV cache layout, which the ranks agree at startup: every rank of the deployment sizes its cache to the same units. A group on another host (`rank-1/cuda:0`) needs a deployment of several nodes, whose Storage Servers listen on TCP, with NIXL on both hosts' library search path ([NIXL transfers](nixl.md)). Such a deployment generates text: canvas requests are refused.

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

A `ranks` entry is `[HOST/]DEVICE`. `DEVICE` is one device (`cuda:0`, `cpu`), an inclusive index range (`cuda:0-3`) or a device repeated for CPU workers (`cpu*4`). Without `HOST` the ranks run on the leader's own host, the `host_identity` of the process that reads the file; `rank-1/cuda:0-3` places four ranks on the node whose `uniserve serve` starts with `--host-identity rank-1`.

A block table takes:

| Key | Default | Meaning |
| --- | --- | --- |
| `ranks` | Every rank of the replica | Indices into one replica's ranks; their order assigns each rank its position in the parallel degrees or the unit schedule. A denoiser block's positions run over the Ulysses degree fastest, then the tensor degree, then the all-gather degree, and the pipeline degree slowest: in a block of `tensor_parallel_size = 2` and Ulysses degree 2, positions 0 and 1 are one Ulysses group and positions 0 and 2 one tensor group |
| `tensor_parallel_size`, `pipeline_parallel_size` | `1` | Model-parallel degrees |
| `sequence_parallel` | `{ kind = "local" }` | `{ kind = "ulysses", ulysses_degree = N }` with optional `projection` and `projection_rows`, `{ kind = "allgather", allgather_degree = N }`, or `{ kind = "hybrid", ulysses_degree = N, allgather_degree = M }` |
| `distribution` | Unset | `temporal_units` runs its own copy of the block on every rank and deals independent media units among them; the video decoder and codec always take it |
| `units_per_rank` | `1` | Consecutive units each rank takes under `distribution` |
| `weights` | `device` | `host` keeps the dense projection weights of the block's repeated layer stack in page-locked host storage and serves it without CUDA graphs |

Without a distribution, a block's ranks number the product of its degrees. `configs/` holds the published deployments, and the [FastH3 guide](minimax_h3/minimax_h3.md#deployments) lists the FastH3 files.

A text model's `model` block splits the model along one axis. `tensor_parallel_size` splits every layer's heads and projections. `pipeline_parallel_size` gives each rank, in the block's rank order, an equal contiguous share of the decoder layers, each share holding as many attention layers as the others, and every rank keeps the embedding, the head, the drafter and the vision tower whole; the stages of a step run one after another, the last stage's final rows reaching every stage, and a group's stages lie on one host. A `ulysses` sequence split gives each rank a share of every batch's rows and of the attention and key/value heads, which its degree must divide, under either projection; it serves neither a model whose layers keep a recurrent state nor speculative decoding, and `projection_rows` is refused. A block with two degrees above one, the `allgather` and `hybrid` kinds, and serving groups of one deployment that split the model differently are refused at startup.
