# Configuration

A UniServe server is configured by one tree of serving options and, for a placement other than one tensor-parallel worker group per scheduler, a deployment file. The tree's top level holds the options most deployments state; the rest sit in `*_config` tables, one per part of the system they configure.

## Command line

`uniserve serve` reads options from three places, each overriding the last:

1. **`--config FILE`**: a TOML file of the tree.
2. **Flags** for the most common options, each a shortcut for one path of the tree.
3. **`--set PATH=VALUE`**, repeatable, reaching any option by its path.

Options none of them states keep their defaults. `--print-config` prints the resolved tree, defaults included, and exits; options unset by default are left out of the printout.

```bash
uniserve serve "$H3_MODEL" --worker-groups configs/minimax_h3/ulysses4.toml \
  --max-running-requests 2 \
  --set cuda_graph_config.policy=off \
  --set video_config.text_capacities=1024,10240

uniserve serve --config fasth3.toml --set quantization_config.mode=performance
uniserve serve --config fasth3.toml --print-config
```

| Flag | Path |
| --- | --- |
| `MODEL` (positional) | `model` |
| `--host`, `--port` | `server_config.host`, `server_config.port` |
| `--served-model-name` | `served_model_name` |
| `--worker-groups` | `worker_groups` |
| `--tensor-parallel-size`, `--data-parallel-size` | `tensor_parallel_size`, `data_parallel_size` |
| `--max-running-requests` | `max_running_requests` |
| `--max-model-len` | `max_model_len` |
| `--mem-fraction-static` | `mem_fraction_static` |
| `--host-identity` | `host_identity` |
| `--nnodes`, `--node-rank`, `--dist-init-addr` | `nnodes`, `node_rank`, `dist_init_addr` |
| `--worker-executable` | `worker_config.executable` |

`--log-level` and `--log-level-http` set the log levels of the `uniserve` process; every subcommand takes them, before or after its name.

A `--set` value is a TOML value (`2`, `true`, `"text"`, `[1024, 4096]`, `{ mode = "balanced" }`) or JSON, and otherwise the text itself, so `--set cuda_graph_config.policy=off` and `--set server_config.host=0.0.0.0` need no quotes. A list may be comma-separated text (`--set video_config.aspect_ratios=16:9,9:16`); quoting is safer than brackets, which the shell may expand. One invocation may set a path several times, the last winning, but not both a table and an option inside it.

Every option is checked by its type wherever it is stated, and an error names its path: an unknown option, a value its option does not take, or zero for a positive count. A duration is whole seconds or a value with units (`500ms`, `30s`, `5m`, `1h`). A byte size is whole bytes or an integer with a decimal (`20MB`) or binary (`512MiB`) unit. Paths in options resolve against the server's working directory.

Each host's Storage Server (`uniserve storage serve`, see `architecture.md`) is started by the host's daemon when the host has none and keeps running across server restarts. It takes its own options when started by hand before the server: `uniserve storage serve --page-pool-gib N` sizes a host page pool whose worker states no size (64 GiB unless stated), `--host-reserve-gib N` sets the host memory a new page pool leaves to other processes (10 GiB unless stated: a pool the host's available memory less the reserve cannot hold is refused), `--transfer-chunk-us N` sets how long one chunk of its Transfer Server's copies lasts at the link's bandwidth (1000 µs unless stated), `--nixl-library PATH` names NIXL's `libnixl_capi.so` (of the NIXL release the build pins, 1.5.1) that the Transfer Server loads for copies from other hosts (the loader's search path finds it unless stated; `UCX_NET_DEVICES` in the server's environment limits UCX to the host's usable RDMA ports), and `--mooncake-master HOST:PORT` with `--mooncake-metadata`, `--mooncake-protocol`, `--mooncake-device`, `--mooncake-hostname`, `--mooncake-segment-bytes` and `--mooncake-library` runs its Mooncake connector, as `kv_cache_config.mooncake` does for a server the deployment starts. `uniserve storage list` prints what it holds, and `uniserve storage retire --deployment NAME` releases the memory of a deployment that is no longer served. A deployment is named by its served model name, so a restarted server finds the memory it left under the same name. Page pools belong to a deployment: each worker maps one per cache group, holding the token count the engine sizes its world's host KV cache at (`kv_cache_config.host_ratio` and `host_gib`), and a pool is allocated and pinned whole when the first worker maps it. A world shares the pools of the previous world whose pages have the same layout; a pool the host cannot hold first takes the place of the deployment's pools no world maps any more. When a deployment claims a host, the server retires every other deployment whose daemon is no longer connected, so a host keeps the weights and caches of the deployments it serves only.

Some tables configure one model family: `chat_config` and `image_fetch_config` chat models (Qwen3, SenseNova, BAGEL, DiffusionGemma); `video_config`, `media_config` and `video_job_config` MiniMax-H3; `readout_config` and `diffusion_generation_config` DiffusionGemma. Startup refuses such a table stated for a checkpoint of another family, naming the table and the served model.

## Configuration file

A configuration file is the tree in TOML, as `--print-config` prints it:

```toml
model = "/workspace/models/FastVideo-FastH3-8-Step-V2"
served_model_name = "FastH3"
worker_groups = "configs/minimax_h3/ulysses4.toml"
max_running_requests = 2

[server_config]
host = "0.0.0.0"

[quantization_config]
mode = "quality"

[cuda_graph_config]
policy = "auto"

[video_config]
text_capacities = [1024, 10240]

[video_job_config]
retention = "30m"
```

## Options

### Top level

| Option | Default | Meaning |
| --- | --- | --- |
| `model` | Required | Local model directory or Hugging Face repository |
| `served_model_name` | `model` | The single model name the API serves |
| `max_model_len` | The checkpoint's `max_position_embeddings`, 8192 when it declares none; 16384 prompt tokens for a MiniMax-H3 pipeline | Longest sequence a request may reach, in tokens |
| `dtype` | `"bfloat16"` | Element type of parameters and activations: `bfloat16`, `float16` or `float32` |
| `quantization_config` | `{}` | Precision policy: `mode` (`quality`, `balanced`, `performance`, `maximum`) and per-component selections under `components`; empty selects the model's default |
| `mm_processor_kwargs` | `{}` | Options of the model's multimodal processor; MiniMax-H3 takes `reference_image_short_edge` and `enlarge_references` ([FastH3 guide](minimax_h3/minimax_h3.md#startup-and-capacity)) |
| `attention_backend` | `"auto"` | Attention provider: `auto` selects a native kernel per call; `flashinfer`, `trtllm`, `flash_attn_4`, `torch`, `prefix_block`, `flash_attn` or `sgl_kernel` serves every call |
| `max_running_requests` | `128` | Concurrently resident requests, clamped to what the workers hold |
| `max_num_batched_tokens` | `8192` | Token work admitted in one scheduling step; token models only |
| `mem_fraction_static` | `0.7` | The deployment's share of each device's memory for weights, static allocations, graph pools and the KV cache, in (0, 1]: the quota `⌊total × fraction⌋` of the GPU's memory ledger, which both worlds of a configuration change and the host's Transfer Server share (`docs/admin.md`); a deployment group's `mem_fraction_static` overrides it |
| `tensor_parallel_size` | `1` | Tensor-parallel workers of each scheduler's group without `worker_groups` |
| `data_parallel_size` | `1` | DP ranks, each a scheduler with its own KV cache and worker groups; requests go to the scheduler with the fewest in flight. With `worker_groups`, the file's model groups form the schedulers as equal consecutive blocks |
| `worker_groups` | Unset | [Deployment file](#deployment-files) placing worker groups and their components |
| `host_identity` | `"localhost"` | This node's host name; its daemon runs the ranks placed on this host, and a deployment rank entry without a host names it |
| `nnodes` | `1` | Nodes of the deployment, each running one `uniserve serve` |
| `node_rank` | `0` | This node's rank. The leader, rank 0, serves the API and runs the engine, and requires `model`; every other node runs only its daemon, which joins the leader's |
| `dist_init_addr` | Unset | The leader's `HOST:PORT`, which the other nodes' daemons join at and the leader binds on every interface; required with several nodes |

### `server_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `host` | `"127.0.0.1"` | TCP address the server listens on |
| `port` | `8000` | TCP port the server listens on |
| `uds` | Unset | Unix domain socket to listen on instead of TCP; a stale socket file is replaced and the socket file is removed at shutdown |
| `api_key` | Unset | Bearer token every route but `/health` and `/metrics` requires; `UNISERVE_API_KEY` supplies it when unset or blank. Never printed |
| `request_timeout` | Unset | Time until a response head is sent, answered with 504 when it expires; streamed bodies are not bounded |
| `max_concurrent_requests` | Unset | In-flight chat, image generation, System One and synchronous video requests above which requests are shed with 503 |
| `shutdown_timeout` | `"30s"` | Longest drain of in-flight requests at shutdown; `0` stops at once |
| `request_id_headers` | `false` | Add an `X-Request-Id` response header with the ID the request ran under |

### `load_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `format` | `"auto"` | Checkpoint reader: `auto`, `safetensors`, `pt`, `layered` (one layer at a time) or `dummy` (no weights read) |
| `download_dir` | Hugging Face default | Cache root for repository models |
| `threads` | Loader default | Concurrent checkpoint file readers |
| `weights` | unset | A plugin's weight source (`scheme://...`, `docs/plugins.md`) that streams the checkpoint's tensors in place of its files; the model's configuration and tokenizer stay the checkpoint's. An in-place weight update from a source sets it |
| `parallel_degrees` | every degree up to the GPUs a worker sees | The tensor- and expert-parallel degrees the deployment's configurations use. Retained weight memory is allocated in chunks cut where any of these degrees divides a logical tensor, so a world of another of them maps the chunks it shares with the world before it on the same GPU instead of copying them (`docs/architecture.md`, Weights). State them when the deployment may grow beyond the GPUs its workers see |

### `kv_cache_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | Sized from memory | Capacity in tokens |
| `block_size` | Largest supported up to 64 | Tokens per page of the cache group with the widest token rows, a power of two |
| `dtype` | Each layer's compute type | `bfloat16`, `float16`, `float32` or `float8_e4m3fn` |
| `prefix_caching` | `true` | Reuse and retain cached prompt prefixes across requests |
| `host_tier` | `true` | With `prefix_caching`, offload prompt pages to the host KV cache of each host's Storage Server and load them back when a later prompt's prefix continues past the device cache's match: every published page of a full-attention cache group, and a sliding-window group's window before each aligned boundary (`boundary_interval`) |
| `host_ratio` | `2.0` | With `host_tier`, the host KV cache's token count over the device KV cache's, the same on every rank of a world (as SGLang's `--hicache-ratio`): each cache group's pool holds that many tokens' pages. Each host's Storage Server lowers the count for the whole world to what four fifths of its available memory hold (bounded by its memory cgroup limit, counting the deployment's pools no world maps, less what the pools have yet to allocate and its reserve) over the bytes a token takes on all of the world's ranks on the host |
| `host_gib` | None | With `host_tier`, the deployment's host KV cache on each host, in GiB, in place of `host_ratio` (as SGLang's `--hicache-size`): every rank of a world holds as many tokens as this holds over the bytes a token takes on all of the world's ranks on the host, the least over the hosts. The host's Storage Server refuses a pool its available memory cannot hold |
| `mooncake` | None | With `host_tier`, keep every page of the host KV cache in Mooncake, the KV storage, and prefetch hits of at least 256 tokens back into the host KV cache at admission: an object with `master` (`host:port`, required), `metadata_server` (`"P2PHANDSHAKE"`), `protocol` (`"tcp"` or `"rdma"`), `device` (RDMA devices, automatic when empty), `global_segment_gib` (host memory each host contributes to Mooncake, `0`) and `library` (`libmooncake_store.so`, else `MOONCAKE_STORE_LIBRARY` or the loader's search path). It applies to the Storage Servers the deployment starts |
| `boundary_interval` | `64` | With `host_tier`, the pages of the largest block size between the aligned boundaries a long prompt offloads its sliding windows before; a prompt also offloads them before the last boundary a reissue of it reuses and before its last full page. Larger intervals offload fewer windows, and a prompt that continues one offloaded earlier reuses less of it |
| `state_interval` | `256` | For a model that keeps a recurrent state per request (Qwen3.5), the decoded tokens between two checkpoints of a request's state (as SGLang's `--mamba-track-interval`); a prompt call checkpoints the last page boundary it reaches whatever the interval. A request resumed by its tokens (a migration, a world change, a prefix reuse) recomputes at most this many decoded tokens; a checkpoint occupies the units of every recurrent layer's state |

### `scheduler_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"fcfs"` | Waiting-queue order: `fcfs` or `priority` |
| `max_queued_requests` | Unbounded | Waiting requests, and finished ones with undelivered events, before a submission is refused as overloaded |
| `chunked_prefill_size` | `8192` | Prompt tokens one request prefills in one step; token models only |
| `max_batch` | `128` | Calls assembled into one forward batch, clamped to what the workers accept |

### `speculative_config`

Speculative decoding with the model's own drafter, the multi-token prediction layer of a Qwen3.5 checkpoint (`mtp_num_hidden_layers`); token models only. Every step, each decoding request's verification evaluates its committed token followed by the draft tokens the drafter proposed after its last step and commits the longest prefix of drafts the model's own selections agree with, plus the model's token after it, so a request generates the tokens it would without speculation, one or more per step. The scheduler chooses each request's draft count every step: the drafts proposed, at most `num_draft_tokens` and one fewer than the tokens the request may still generate. The drafter keeps a cache of its own per request, the draft cache: optional state held in its own group of the KV cache, published, reused and offloaded to the host KV cache as the other groups' pages are, and released from a running request when another request's required pages need the units. A request without its draft cache (a prompt prefix reused without it, a request a world change moved from a world that did not speculate, a draft cache released for another request) verifies no drafts, a verification of one token, and decodes as without speculation; a refill then evaluates the model again over the positions before the draft cache entries the request wrote itself, in chunks of the token budget a prefill step leaves, when it is worth its cost: when the request's remaining tokens times `1 - 1 / g` exceed the refill's tokens times `k`, both sides in decode steps, with `g` the speedup speculation gives a request and `k` the time of evaluating one context token. Each scheduler measures both from what its world executes, as averages over about the last sixteen observations: a verification carrying drafts commits `s` tokens, the workers' decoding step takes `v` with drafts and `d` without, and a prompt step `p` per token; speculating, a request takes `v / s` per token in place of `d`, so `g = s * d / v` and `k = p / d`, and while every step has carried drafts `d` is `v`. A world whose verification steps cost as many decode steps as they commit tokens (`g` at most `1`) refills nothing; until it has measured what it needs, a world refills, which measures it. A new request writes its draft cache while it prefills and speculates from its first decoding step. Verification and the drafter run eagerly: a model that keeps a recurrent state per request (Qwen3.5) serves eagerly in any case.

| Option | Default | Meaning |
| --- | --- | --- |
| `num_draft_tokens` | `0` | The most draft tokens a verification carries per request; `0` does not speculate. A world that speculates keeps the draft cache in its pool and runs the drafter, so setting or changing it is a world change (`PUT /admin/config`) |
| `expected_speedup` | Measured | The refill policy's speedup `g`, stated in place of the measured one, at least `1`; applied in place by `PUT /admin/config` |
| `refill_token_cost` | Measured | The refill policy's time `k` of evaluating one context token in a refill, in decode steps, stated in place of the measured one, not negative; applied in place by `PUT /admin/config` |

### `migration_config`

A token request that generates text moves to the engine that serves next when its world parks for a world change or a rollback, or loses the workers that hold it; it continues after the last token it delivered, with the same output and usage.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_migrations` | `3` | Migrations caused by lost workers, a lost engine process or a failed world on trial that a request survives before it fails; migrations of a world change or a rollback are not counted |

### `update_config`

How the deployment moves to a configuration that a new world executes (`PUT /admin/config`, [admin API](admin.md#configuration-changes-and-worlds)). An option that an update's own `set` states governs that update alone; the configuration keeps its own.

| Option | Default | Meaning |
| --- | --- | --- |
| `previous_world` | `"parked"` | What the previous world keeps while its successor is on trial: `parked`, its processes, communicators, graph memory and weights, so that a rollback resumes it at once; `weights`, its weights alone, its processes exiting at the promotion, so that a rollback starts it again on them; `host`, as `weights` with the weights its successor does not map moved to host memory through each host's Transfer Server, so that the successor's cache grows into their device memory too, and a rollback returns them to the GPUs before it starts the previous world again. A host whose Storage Server runs no Transfer Server, or that cannot hold them under its host-memory rule, keeps them on its GPUs, and the trial's status reports the previous world kept as `weights` with each host's reason (`host_refused`) |
| `preempt_after` | Unset | How long the serving world's requests that hold the tail of the KV cache a successor needs run on before they are recomputed on the cache left; unset lets them run to their end |
| `transfer_slowdown` | `0.1` | How much longer than before the change a serving scheduler's step may take while the successor prepares, as a fraction (`0.1` for 10%), past which the leader halves the byte budgets of the links the change's transfers use on every node, each second while it lasts, and restores them once every scheduler's steps are back within it; the default is measured on GB200 (below) |
| `eager_prepare` | `false` | Whether the serving world runs its forwards eagerly while its successor prepares, giving its CUDA graph memory (10% of each GPU) to the successor's captures: it steps slower meanwhile, and replays its graphs again if it serves on or a rollback resumes it. A world whose computations serve by their graphs alone (image, video and diffusion language models) refuses it |
| `rollback.failure_rate` | Unset | The share of the requests that end during the trial, in (0, 1], past which a trial whose requests failed rolls back |
| `rollback.step_time_regression` | Unset | How much longer than the previous world's average step before its promotion the successor's may take, as a fraction (`0.2` for 20%), before the trial rolls back |
| `rollback.throughput_regression` | Unset | How much lower than the previous world's token throughput before its promotion the successor's may fall, as a fraction, before the trial rolls back |
| `rollback.window` | Unset | The span step time and throughput are averaged over, before the promotion and during the trial; required with either regression |

A trial also rolls back, whatever these state, on a fault of the successor: a worker error that leaves its workers unusable (an illegal memory access, a worker process's exit), a stalled job, non-finite logits, or its engine process's exit. None of the thresholds has a default: each is measured on the deployment's own workload.

`transfer_slowdown` compares each scheduler's steps with its own steps of the same size and execution before the preparation (`crates/daemon/src/interference.rs`). A step's time is a once-a-second report's host-observed batch time over its batches, the step time the rollback thresholds read; a step's size is the power-of-two range of the report's query tokens per batch, since a step computing more tokens takes longer whatever the transfers do. A scheduler's baseline in a range is its batch time per batch over the minute before the preparation, once its reports there hold 20 batches; a report in a range without a baseline decides nothing. With `eager_prepare` the serving world's eager steps are slower than its graphs' whatever the transfers do, so its baseline holds only the reports of eager steps alone, those that arrive more than a second after the switch and before the preparation, and the successor starts once each scheduler made two of them with steps, about 2 to 3 s after the switch. A scheduler that runs no step for 2.5 s has nothing to protect: the successor starts without its baseline, the change's progress notes it (`GET /admin/config`), and its reports decide nothing. The change's report gives the wait (`eager_baseline_seconds`) and the halvings and restorations the leader sent (`transfer_halvings`, `transfer_restorations`). The halving takes every link a task of the deployment copies over at that moment (a deployment changes one world at a time), and the restoration every link a halving reduced, which the Transfer Server grows back by 1/8 of its bandwidth per second; the end of the preparation restores them too.

The default, 0.1, comes from a measurement on GB200 (`crates/storage-server/examples/transfer_interference.rs`; protocol, raw data and tables in `artifacts/transfer-interference/`): Qwen3-30B-A3B served on one GPU in BF16 under a closed-loop decode load of 64 concurrent requests of 128 prompt and 1024 output tokens (68 query tokens per step, 20.5 ms steps), while the Transfer Server copied into that GPU at a link's whole budget and at 1/4, 1/16 and 1/64 of it, 30 s per point, in three runs, each with a fresh server. The step time is the one the leader reads, per once-a-second report.

| Copies into the serving GPU | Budget 1 | 1/4 | 1/16 | 1/64 |
| --- | --- | --- | --- | --- |
| Within the GPU (1.82 TB/s at budget 1) | ×2.96 | ×1.60 | ×1.14 | ×1.03 |
| From host storage (209 GB/s) | ×1.16 | ×1.02 | ×1.00 | ×1.00 |
| From another GPU of the host (779 GB/s) | ×1.08 | ×1.01 | ×1.00 | ×1.00 |

Each entry is the median over the point's reports of the step time over its run's median without copies (the three runs agree within 0.04). Without copies, a report's step time departs from the median by at most 4.4% in 99% of reports, and one report of 354 exceeds it by more than 10% (15%). Read as the leader decides, one report at a time, a tolerance of 0.1 is exceeded by 0.3% of the reports without copies, by every report of copies within the GPU at budgets down to 1/16 and by 98% of those from host storage at its whole budget, which slow the steps by 14% or more, and by 10% of the reports of copies from another GPU at its whole budget, which slow them by 8%. A tolerance of 0.05 would be exceeded by 0.8% of the reports without copies and by up to 20% of those of copies that slow the steps by 3% or less (within the GPU at 1/64 of its budget), which 0.1 exceeds in at most 5%; one of 0.15 would let copies within the GPU at 1/16 of their budget (14%) through in 59% of the reports. Other hardware, models and loads are measured the same way before the value is changed.

### `placement_config`

`device`, `worker_hosts` and `queue_depth` place ranks without a deployment file, so each is refused together with `worker_groups`, as is `tensor_parallel_size`.

| Option | Default | Meaning |
| --- | --- | --- |
| `device` | `"cuda"` | Device of every rank; `cuda` numbers each host's GPUs from zero |
| `worker_hosts` | `[]` | Hosts after this one the ranks spread across, in equal blocks |
| `queue_depth` | `2` | Batches in flight per rank; a deployment group states its own |
| `transfer` | Derived per edge | Product transport per worker edge: `source[:rank]->destination[:rank]=mechanisms`, comma-separated |

### `moe_config`

[Expert parallelism](expert_parallelism.md) describes each exchange.

| Option | Default | Meaning |
| --- | --- | --- |
| `expert_parallel` | `false` | Shard routed experts across the schedulers, the tensor-parallel ranks of a single scheduler's one worker group, or the deployment's expert workers |
| `exchange` | `"alltoall"` | `alltoall`, `megamoe`, `deepep` or `dwdp` |
| `microbatches` | `1` | Microbatches overlapping attention with dedicated expert workers |

### `cuda_graph_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"auto"` | Whether a worker group runs every call with CUDA graphs or every call eagerly, chosen when its workers load and never mixed: `auto` captures graphs unless serving does not capture the attention provider's calls or a module's (the gated delta attention of Qwen3.5 runs a host loop over its chunks), and then serves eagerly; `full` requires graphs; `off` serves every call eagerly (CPU workers always do). Under graphs the text buckets hold every decode and prefill batch the worker admits; prefill and image denoising are always captured, and a request whose shapes no captured graph holds is refused with the reason, never run eagerly: an image vision encoder must pack its images into fixed slots, and a video denoiser serving `ref2va` serves only with `off` |
| `decode_batch_sizes` | Worker catalog | Decode batch sizes captured, strictly increasing; the largest decode batch the worker admits is captured besides them |
| `prefill_token_sizes` | Worker catalog | Prefill token counts captured, strictly increasing; the batch token bound is captured besides them, for every row bucket up to the largest batch the worker admits |
| `flow_shapes` | The model's default image size and its transpose | Distinct `HEIGHTxWIDTH` image sizes captured. Under graphs an image model generates only these sizes, encodes only the generated images of these sizes into latents (input images whose latent size is not among them are refused), and decodes only these sizes |

### `flashinfer_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `workspace_size` | `"512MiB"` | Workspace per rank |
| `use_tensor_core` | Backend decides | Decode attention on tensor cores |
| `decode_backend`, `prefill_backend` | `"fa2"`, `"auto"` | Kernels: `auto`, `fa2` or `fa3` |
| `decode_split_tile_size`, `prefill_split_tile_size` | FlashInfer default | Split-KV tile sizes |
| `split_kv` | `true` | Split a KV sequence across thread blocks |

### `worker_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `executable` | The daemon's own executable when it is installed in a Python environment, else the active virtual environment's `uniserve` | The `uniserve` executable the workers run as (`uniserve worker`), and that starts a host's Storage Server; it runs workers in the Python environment it is installed in |
| `watchdog_timeout` | `"5m"` | Time a batch may run without progress of its workers before the worker group is replaced; a worker reports a stalled job after half of it |
| `mesh` | Unset | `tower=gen:<device>` places the generation tower on another device than the worker's |
| `distributed_backend` | Torch default | Torch process-group backend for the workers' collectives |
| `standby` | `false` | Keep a standby beside every worker group. The standby maps the group's retained weights and cache, captures its graphs on warmup scratch units and parks; when the group fails, it is promoted in the group's place instead of a relaunch, and a new standby starts. A group whose ranks all run on the leader's host and that shares no expert world keeps a standby rank group of its own. The groups of an expert world, and a group on several nodes, keep a standby world: a second launch of every group of their failure domain, which forms communicators of its own and is promoted as a whole when any group of the domain fails. Each standby rank holds a CUDA context, graph pools and workspaces of its own, which the device memory `mem_fraction_static` leaves must hold |
| `standby_memory` | `auto` | How a parked standby (a group's or a failure domain's standby world) keeps the device memory of its graph pools, operator scratch, expert all-to-all workspaces and NCCL communicators: `resident`; `suspended`, released on the device and restored at the promotion, before its graphs run (the graph memory is copied to host memory and back with its contents at unchanged addresses; the `alltoall` exchange's MNNVL workspace is unmapped and mapped again at its addresses by every rank of the world together; NCCL communicators are suspended and resumed, and such a standby captures their collectives without NCCL graph buffer registration; the `deepep` exchange's windows and the colocated `megamoe` exchange's NVSHMEM heap, which cannot keep their addresses, are freed, and the promotion creates them again and captures the graphs that run expert steps anew); or `auto`, resident while every device keeps beside it a tenth of its memory, the share a serving worker leaves for graphs and activations, and suspended otherwise. The ranks of a world or group take one mode: one rank's crowded device suspends them all. A standby that lacks that share even suspended fails, and none is kept |

A worker compiles the kernels its model runs that its machine has not cached yet, on first use. The compilations run up to `MAX_JOBS` parallel jobs when the worker's environment, its host daemon's, sets it. Otherwise each worker takes an equal share, among the ranks of its launch on its host, of half of the host's available CPUs and of the host memory available as it starts, at 6.1 GiB per job (the peak of one compiler process of FlashInfer's fused-MoE kernels), and at least one job; its log states the number.

### `profiling_config`

The workers' always-on profiling: the fence probes at the entry and exit of the model's components, exported as per-step component durations (`uniserve:worker_fence_step_seconds`, [execution profiling](profiling.md#fence-probes)). Profiling sessions take their options with their request (`POST /admin/profile`).

| Option | Default | Meaning |
| --- | --- | --- |
| `fence_probes` | `true` | Whether the probes record; a change switches every worker's probes in place, at the engine's next step boundary |
| `fence_components` | `["*.Transformer"]` (each model's decoder stack, within the always-on budget `docs/profiling.md` states) | Name patterns (`*` matches any run) of the components whose module instances carry probes, a component being the `<component>` of the function its module class declares (`uniserve/numerics.py`); the probes are captured into a world's CUDA graphs, so a change takes a new world. Empty for no probe |

### `logging_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `log_requests` | `false` | Log a summary line for each completed request |
| `log_stats` | `true` | Log engine statistics periodically |

### `chat_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `template` | The checkpoint's | Chat template, as a file path or the template itself |
| `template_kwargs` | Unset | Keyword arguments bound into every chat-template render |
| `template_content_format` | `"auto"` | How message content renders: `auto`, `string` or `openai` |
| `reasoning_parser` | `"auto"` | `auto` splits `reasoning_content` from `content`; `none` streams reasoning delimiters as content |

### `image_fetch_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `timeout` | `"20s"` | Time to fetch one `http(s)` image URL, redirects and body included |
| `max_bytes` | `20000000` | Largest input image, fetched or `data:` |
| `allow_private` | `false` | Allow image URLs resolving to loopback, private, link-local, unique-local or cloud metadata addresses |

### `video_config`

The [FastH3 guide](minimax_h3/minimax_h3.md#startup-and-capacity) explains how these options size startup and capacity.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_seconds` | `15` | Longest admitted clip, 4 to 15 seconds |
| `resolutions` | `["768p"]` | Resolution classes startup prepares, each at every aspect ratio |
| `aspect_ratios` | `["16:9", "9:16"]` | Aspect ratios startup prepares |
| `text_capacities` | `1024`, then steps of 2048 | Prompt capacities below `max_model_len`, strictly increasing |
| `max_condition_rows` | Largest condition set | Denoiser rows one request's conditions may take |

### `media_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `directory` | Unset | Directory `file://` condition media resolves under; `file://` is refused when unset |
| `allow_remote` | `true` | Fetch `http(s)://` condition media |
| `max_request_bytes` | `"256MiB"` | Condition media one request may carry in total; the video routes' body limit follows it |
| `ffprobe`, `ffmpeg` | `"ffprobe"`, `"ffmpeg"` | Executables that probe condition media and decode reference videos |

### `video_job_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `retention` | `"1h"` | Time a finished asynchronous job stays retrievable |
| `max_bytes` | `"1GiB"` | Generated video the asynchronous jobs keep in shared memory |

### `readout_config`

The [DiffusionGemma serving guide](diffusion_gemma/serving.md) explains these options.

| Option | Default | Meaning |
| --- | --- | --- |
| `layout` | `"joint"` | `joint` packs System One questions into shared canvases; `independent` gives each its own prompt and canvas |
| `canvas` | `"full"` | Canvas length: `full`, `compact`, or a positive multiple of 16 |
| `candidates` | `"variants"` | `variants` sums supported answer spellings; `primary` reads the space-prefixed one |

### `diffusion_generation_config`

DiffusionGemma's block-diffusion sampling of every generated reply; each option stated replaces the checkpoint's `generation_config.json` value of the same name: `max_denoising_steps`, `entropy_bound`, `t_min`, `t_max`, `confidence_threshold` and `stability_threshold`.

## Deployment files

A deployment file, the `worker_groups` option, places worker groups on devices and hosts and names the components each runs. It is TOML: one `[[worker_groups]]` table per group and one `[worker_groups.components.<name>]` table per component of the group, as in `configs/minimax_h3/fasth3-ulysses4x2.toml`:

```toml
# FastH3 on eight GPUs of one host: two Ulysses4 denoiser replicas, each with
# its own TP4 text encoder.

[[worker_groups]]
id = "text"
replicas = 2
ranks = ["cuda:0-7"]
queue_depth = 6
mem_fraction_static = 0.26

[worker_groups.components.text_encoder]
tensor_parallel_size = 4

[[worker_groups]]
id = "flow"
replicas = 2
ranks = ["cuda:0-7"]
queue_depth = 6
mem_fraction_static = 0.73

[worker_groups.components.transformer]
sequence_parallel = { kind = "ulysses", ulysses_degree = 4 }

[worker_groups.components.video_decoder]

[worker_groups.components.audio_decoder]
distribution = "temporal_units"

[worker_groups.components.latent_encoder]
distribution = "temporal_units"

[[worker_groups]]
id = "encoder"
replicas = 2
ranks = ["cpu*8"]
queue_depth = 6

[worker_groups.components.video_codec]

[[worker_groups]]
id = "reader"
ranks = ["cpu"]
queue_depth = 6

[worker_groups.components.media_reader]

[[worker_groups]]
id = "muxer"
ranks = ["cpu"]
queue_depth = 12

[worker_groups.components.muxer]
```

A `[[worker_groups]]` table takes:

| Key | Default | Meaning |
| --- | --- | --- |
| `id` | Required | Group name; replica `i` of a replicated group is `<id>-<i>` |
| `role` | `model` | `model` for a group serving model components; `prefill` and `decode` for a prefill group and a decode group of the text model (below); `experts` for dedicated expert ranks, which take no components |
| `replicas` | `1` | Copies of the group, each over the next equal block of `ranks`. The copies' components are replicas of the same components under one scheduler, which binds each request to one copy |
| `ranks` | Required | The group's ranks, every replica's together, in rank order |
| `queue_depth` | Required | Batches in flight per rank. A rank that keeps video requests resident takes three positions per request, so such a group holds up to `queue_depth / 3` requests at once; a group serving several replicas, such as a shared text encoder, needs positions for all of them |
| `mem_fraction_static` | Top-level `mem_fraction_static` | Each rank's device-memory share for this group |

A `prefill` group prefills prompts only and a `decode` group decodes only; a deployment configures both or neither, and each places the `model` component alone. A request prefills on a prefill group; once its prompt is complete and its first token sampled, the prefill group exports its KV cache, a decode group installs it into the same cache units (`CacheExport` and `CacheInstall` transfer calls), and the request decodes there. Such a deployment generates text: image generation and canvas requests are refused.

A `ranks` entry is `[HOST/]DEVICE`. `DEVICE` is one device (`cuda:0`, `cpu`), an inclusive index range (`cuda:0-3`) or a device repeated for CPU workers (`cpu*4`). Without `HOST` the ranks run on the leader's own host, the `host_identity` of the process that reads the file; `rank-1/cuda:0-3` places four ranks on the node whose `uniserve serve` starts with `--host-identity rank-1`.

A component table takes:

| Key | Default | Meaning |
| --- | --- | --- |
| `ranks` | Every rank of the replica | Indices into one replica's ranks; their order assigns each rank its position in the parallel degrees or the unit schedule |
| `tensor_parallel_size`, `pipeline_parallel_size` | `1` | Model-parallel degrees |
| `sequence_parallel` | `{ kind = "local" }` | `{ kind = "ulysses", ulysses_degree = N }` with optional `projection` and `projection_rows`, `{ kind = "allgather", allgather_degree = N }`, or `{ kind = "hybrid", ulysses_degree = N, allgather_degree = M }` |
| `distribution` | Unset | `temporal_units` runs its own copy of the component on every rank and deals independent media units among them; the video decoder and codec always take it |
| `units_per_rank` | `1` | Consecutive units each rank takes under `distribution` |
| `weights` | `device` | `host` keeps the dense projection weights of the component's repeated layer stack in page-locked host memory and serves it without CUDA graphs |

Without a distribution, a component's ranks number the product of its degrees. `configs/` holds the published deployments, and the [FastH3 guide](minimax_h3/minimax_h3.md#deployments) lists the FastH3 files.
