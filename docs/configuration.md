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

Each host's Storage Server (`uniserve storage serve`, see `architecture.md`) is started by the host's daemon when the host has none and keeps running across server restarts. It takes its own options when started by hand before the server: `uniserve storage serve --page-pool-gib N` sizes each host page pool (64 GiB unless stated; a pool is one per KV head range of the host's workers, and is allocated and pinned whole when the first worker maps it), and `--mooncake-master HOST:PORT` with `--mooncake-metadata`, `--mooncake-protocol`, `--mooncake-device`, `--mooncake-hostname`, `--mooncake-segment-bytes` and `--mooncake-library` runs its Mooncake connector, as `kv_cache_config.mooncake` does for a server the deployment starts. `uniserve storage list` prints what it holds, and `uniserve storage retire --deployment NAME` releases the memory of a deployment that is no longer served. A deployment is named by its served model name, so a restarted server finds the memory it left under the same name.

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
| `mem_fraction_static` | `0.7` | Each rank's share of its device's memory for weights, static allocations and the KV cache, in (0, 1]; a deployment group's `mem_fraction_static` overrides it |
| `tensor_parallel_size` | `1` | Tensor-parallel workers of each scheduler's group without `worker_groups` |
| `data_parallel_size` | `1` | DP ranks, each a scheduler with its own KV cache and worker groups; requests go to the scheduler with the fewest in flight. With `worker_groups`, the file's model groups form the schedulers as equal consecutive blocks |
| `worker_groups` | Unset | [Deployment file](#deployment-files) placing worker groups and their components |
| `host_identity` | `"localhost"` | This node's host name; its daemon runs the ranks placed on this host, and a deployment rank entry without a host names it |
| `nnodes` | `1` | Nodes of the deployment, each running one `uniserve serve` |
| `node_rank` | `0` | This node's rank. The leader, rank 0, serves the API and runs the engine, and requires `model`; every other node runs only its daemon, which joins the leader's |
| `dist_init_addr` | Unset | The leader's `HOST:PORT`, which the other nodes' daemons join at and the leader binds on every interface; required with several nodes |
| `lanes` | Unset, no SM partition | Execution lanes, each `{ lane_id, sm_budget, domains }` with optional `max_batch_calls`, `max_batch_tokens` and `max_inflight`; `domains` are `prefill`, `decode` and `flow` |

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

### `kv_cache_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | Sized from memory | Capacity in tokens |
| `block_size` | Largest supported up to 64 | Tokens per page of the cache group with the widest token rows, a power of two |
| `dtype` | Each layer's compute type | `bfloat16`, `float16`, `float32` or `float8_e4m3fn` |
| `prefix_caching` | `true` | Reuse and retain cached prompt prefixes across requests |
| `host_tier` | `true` | With `prefix_caching`, copy full prompt pages to the host page tier of each host's Storage Server and read them back when a later prompt's prefix continues past the device cache's match; serves worker groups whose cache is one full-attention group |
| `mooncake` | None | With `host_tier`, keep every page of the host page tier in Mooncake, the distributed tier, and prefetch hits of at least 256 tokens back into the host page tier at admission: an object with `master` (`host:port`, required), `metadata_server` (`"P2PHANDSHAKE"`), `protocol` (`"tcp"` or `"rdma"`), `device` (RDMA devices, automatic when empty), `global_segment_gib` (host memory each host contributes to Mooncake, `0`) and `library` (`libmooncake_store.so`, else `MOONCAKE_STORE_LIBRARY` or the loader's search path). It applies to the Storage Servers the deployment starts |

### `scheduler_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"fcfs"` | Waiting-queue order: `fcfs` or `priority` |
| `max_queued_requests` | Unbounded | Waiting requests, and finished ones with undelivered events, before a submission is refused as overloaded |
| `chunked_prefill_size` | `8192` | Prompt tokens one request prefills in one step; token models only |
| `max_batch` | `128` | Calls assembled into one forward batch, clamped to what the workers accept |

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
| `expert_parallel` | `false` | Shard routed experts across the schedulers or the deployment's expert workers |
| `exchange` | `"alltoall"` | `alltoall`, `megamoe`, `deepep` or `dwdp` |
| `microbatches` | `1` | Microbatches overlapping attention with dedicated expert workers |

### `cuda_graph_config`

| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"auto"` | `auto` captures graphs unless serving does not capture the attention provider's calls, `full` requires them, `off` serves every call eagerly |
| `capture_prefill` | `true` | Replay a captured graph for every prefill call |
| `capture_flow` | `true` | Capture and replay image denoising calls |
| `decode_batch_sizes` | Worker catalog | Decode batch sizes captured, strictly increasing |
| `prefill_token_sizes` | Worker catalog | Prefill token counts captured, strictly increasing |
| `flow_batch_sizes` | `[1, 2, 3, 4]` | Image denoising batch sizes captured, strictly increasing |
| `flow_shapes` | `["1152x2048", "2048x1152"]` | Distinct `HEIGHTxWIDTH` image sizes captured |

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
| `watchdog_timeout` | `"5m"` | Time a batch may run without progress of its workers before the worker group is replaced; a worker reports a stalled lane after half of it |
| `mesh` | Unset | `tower=gen:<device>` places the generation tower on another device than the worker's |
| `distributed_backend` | Torch default | Torch process-group backend for the workers' collectives |
| `standby` | `false` | Keep a standby rank group beside each worker group whose ranks all run on the leader's host and that shares no expert world. The standby maps the group's retained weights and cache, captures its graphs on warmup scratch units and parks; when the group fails, it is promoted in the group's place instead of a relaunch, and a new standby starts. Each standby rank holds a CUDA context, graph pools and workspaces of its own, which the device memory `mem_fraction_static` leaves must hold |

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
| `role` | `model` | `model` for a group serving model components; `experts` for dedicated expert ranks, which take no components |
| `replicas` | `1` | Copies of the group, each over the next equal block of `ranks`. The copies' components are replicas of the same components under one scheduler, which binds each request to one copy |
| `ranks` | Required | The group's ranks, every replica's together, in rank order |
| `queue_depth` | Required | Batches in flight per rank. A rank that keeps video requests resident takes three positions per request, so such a group holds up to `queue_depth / 3` requests at once; a group serving several replicas, such as a shared text encoder, needs positions for all of them |
| `mem_fraction_static` | Top-level `mem_fraction_static` | Each rank's device-memory share for this group |

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
