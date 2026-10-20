# Architecture

UniServe is a Rust serving system with a Python numerical library. Rust owns every host responsibility: HTTP, scheduling, the worker processes and the channel to them, request state, storage, CUDA streams and graph policy, transfers, collective orchestration, media containers and observability. The Python package `uniserve` owns numerical computation: model definitions, layers, kernel bindings, samplers and solvers. The two languages meet only in function calls through TVM-FFI.

This document describes how a deployment is organized, how a worker divides its work between Rust and Python, and how a request travels through it. A host is one machine; a worker is one process that executes batches, holding one device or none; a worker group is the workers launched, connected and replaced together; a rank is a worker's index within its group; an engine core is one data-parallel copy of scheduling.

## Principles

1. **Rust is the control plane.** Every decision and every piece of request state is Rust's: requests, slots, progress, sampling policy, random streams, storage assignment, graph policy, stream ordering, transfers and failure handling.
2. **Python is stateless.** Python provides resources and numerical execution on demand and keeps nothing about a request between calls.
3. **The languages meet only in function calls.** A call passes tensors, scalars, flat lists of integers or floats, strings and opaque handles, and returns the same. Neither side depends on a memory layout, record or encoding that the other defines.
4. **Each behavior is implemented once.** A forward's computation is one method of a runner, used by serving, by graph capture and by Python callers alike. A rule both languages need is written in one of them and called from the other.
5. **Python never runs concurrently with itself.** In a worker at most one thread executes Python at any instant, so no thread waits for the GIL.

## Processes

| Process | Language | Role |
| --- | --- | --- |
| `uniserve` (head) | Rust | HTTP API, model profile, tokenization, request lowering and response assembly (`uniserve-server`); scheduling, storage assignment and the worker groups (`uniserve-engine`) |
| `uniserve-launcher` | Rust | Starts and supervises the workers placed on a host other than the head's |
| `uniserve-dynamo` | Rust | An experimental head for FastH3 that embeds the engine and the serving runtime in a Dynamo worker; Dynamo's frontend replaces the HTTP API |
| `uniserve-worker` | Rust, hosting a Python interpreter | One per GPU worker and per CPU worker; executes the batches the engine sends |
| `uniserve-storage-server` | Rust | One per host and serving user; keeps the memory that outlives the head and the workers (see Memory and storage) |

The head contains no Python. A worker starts as `uniserve-worker`, installed into the Python environment's `bin/` beside `uniserve`. The executable finds the environment from its own path, asks the environment's interpreter where its shared `libpython`, `libtvm_ffi` and the worker library `uniserve/_worker` (a TVM-FFI library, not a Python extension module) are, loads the first two into the process's global scope and then the library, the very file Python resolves `uniserve._worker` to, and calls the library's `worker_main` on the process's main thread. `worker_main` spawns the worker's Python thread, which initializes CPython with the environment's interpreter as its executable and no signal handlers, calls `uniserve.worker.attach()` (logging, faulthandler dumps on SIGQUIT and SIGUSR1, and the TVM-FFI global function `uniserve.worker.import_path`, the one function through which Rust reaches the library), releases the GIL and serves Python calls; the main thread runs the control plane. The process exits with the status `worker_main` returns, without Python finalization. Python processes still load the library, for tests, harnesses and the video rules it exports, and in a worker that load returns the instance the executable loaded.

### The channel

The engine and each worker exchange the messages of `uniserve-ipc` (`NewRequest`, `Batch`, `BatchOutput`, `WorkerInfo`) over one stream connection per worker.

- **Connection.** The head listens on one Unix socket path and one TCP port. A worker connects to the head when it starts: through the Unix socket on the head's host, through TCP from any other host. Workers and launchers accept no channel connection; the connections workers accept are their group's peer mesh and its collective store.
- **Registration.** A worker's first frame registers it: worker group, rank, launch generation and build. The head numbers each launch of a worker group, and a relaunch takes the next number; the build is the package version plus a digest of the sources every artifact is compiled from, so the head, launchers and workers of one build carry the same value. The head refuses a registration of another build or of a launch that is not the group's current one, and otherwise answers with the worker's launch descriptor, which carries every other launch value, among them the deployment's name of the worker's host, its identity in transfers.
- **Lifetime.** The same connection then carries requests and responses, and its close is the only death signal of a registered worker: the head treats the worker as dead and the worker exits. Before a worker registers, the head notices the exit of a process it started itself, and a launcher reports the exits of the workers it started.
- **Frames.** A frame is a length followed by one message in a serde binary encoding (MessagePack). Head and workers are built from one workspace, so the message types are the protocol's only definition. A frame carries a message of any size its `u32` length states; no launch setting bounds it further.
- **Waiting.** Each side blocks on its socket together with one local wake descriptor, which its own threads fire for work that did not arrive on the socket.
- **Launchers.** A `uniserve-launcher` launcher's connection carries the same frames: its registration, the head's instructions to reserve, start and stop a group's workers and to terminate, and its reports of reservations and worker exits.
- **Data.** Tensors, prefix cache pages, latents and products move through the transfer mechanisms. Only a host product routed between hosts travels on the channel.

## The planes of a worker

| Plane | Thread | Owner | Blocks only on | Does |
| --- | --- | --- | --- | --- |
| Control | the main thread | `Service`, `Executor`, the Rust half of each runner | Its socket and wake descriptor | Serves the channel, holds request state, admits batches, prepares each forward's inputs, picks its graph, builds its job, applies completions, retires storage |
| Execution | one per lane (`lane-N`) | `Lane`, `Job` | The work on its own stream | Owns the lane's stream: input copies, graph launches, event records and waits, output reads; hands Python work for its stream to the Python thread and waits for it |
| Resource | `python` | `Interpreter` | The interpreter | Executes every Python call, one at a time: loading, execution contexts, captures, eager forwards, releases |

Threads exchange messages only. The control thread submits jobs and receives completions; it calls neither the driver nor Python while it serves, so a slow launch, a driver call that waits on the device, or a stopped lane does not keep it from serving the channel, handling cancellation or reporting status. A lane's execution thread is the only issuer of work on its stream, so the stream's order is that thread's order. A lane posts a job's completion once it has issued the job's work, not once the device has finished it: memory a job used is read, rewritten or freed afterwards only behind something that waited for the device (a read in the job, a stream wait the job recorded, events the job completes after, or the fences that retire storage). Every Python call runs on the Python thread, in arrival order: an execution thread hands it a call and waits for the result; the control thread submits work that owns what it uses and never waits for it; a released Python object is dropped there in queue order, and nobody waits for the drop.

Eager expert forwards with several microbatches are the one exception to a single thread per lane: each forward stays suspended inside Python at its expert dispatch points, so each microbatch has a thread of its own, and those threads call Python themselves as cooperative turns, exactly one at a time, so they never contend for the GIL.

Blocking host work runs on host lanes, threads (`<lane>-<index>`) that each serve one kind of work and hand any Python call to the Python thread like every other thread: media (`uniserve-media`), image preparation (`uniserve-image`), prefix cache imports (`worker-cache-import`), transfer reads (`uniserve-<kind>-read`) and shared-memory retirement (`uniserve-posix-shm-retire`). Signals (`worker-signals`), VMM descriptor grants (`worker-descriptor-grants`) and, during startup, the head watch (`uniserve-head-watch`) are dedicated threads. The Python thread is created by Rust and is Python's main thread; Python installs no signal handlers in a worker.

## How Rust calls Python

Rust reaches the library by import path, for example `uniserve.runner.causal_lm.CausalLMRunner.decode`; there is no table of entry points and no map of named buffers.

- Arguments and results are tensors, scalars, flat lists, strings and opaque handles. A function names every tensor it reads or writes as a parameter.
- Each view of Rust storage is wrapped once with `torch.from_dlpack`; the torch tensor crosses as an opaque Python object, so every call that takes the view receives that same tensor with no per-call conversion. The tensor retains the Rust allocation, and its handle is released on the Python thread with the view's last clone.
- Resident tensors (prefix cache planes, slot tables) are passed once, to a runner's constructor; tensors and sizes that change per forward are parameters of its methods. A tensor only Python reads and writes is a Python intermediate and does not cross.
- A Python object that outlives a call is held by Rust as a handle: a model, an execution context, a runner, a graph. A method does not depend on state an earlier call left in Python, with one exception: a host-planned attention reader's launch plan, which the runner's `plan_*` method builds on the Python thread immediately before the forward, in the same job.
- A Python exception crosses with its kind, message and traceback (`uniserve.errors`).

### Runners

Each capability has one runner in two halves with the same class and file name: `uniserve/runner/<name>.py` and `crates/worker/src/runner/<name>.rs`. The Rust half prepares a forward's inputs, picks its graph, records its job and applies its outputs. The Python half is the forward itself, captured into a graph or run eagerly; a Python caller uses it directly.

| Capability | Runner | Python methods | Request state (Rust memory) |
| --- | --- | --- | --- |
| `CausalLM` | `CausalLMRunner` | `prefill`, `decode`, `verify`, `prompt_logprobs`, the sampling methods, `commit`; `plan_prefill` and `plan_decode` for a host-planned attention reader | prefix cache units, slot tables and lengths, continuation tokens, sampling positions and draws |
| `DiffusionLM` | `DiffusionLMRunner` | `attend`, `readout`, `denoise`; `plan_attention` for a host-planned reader | prefix cache units, each slot's token sequence |
| Text, vision and latent encoders | `EncoderRunner` | `encode`, `position_ids`, `warm`, `tower` (the runner of a nested encoder sharing its context), and for images `prepare_image`, `prepare_resident`, `draw_noise` | products, page-locked prepared images, posterior noise |
| Image and video denoisers | `DenoiserRunner` | `draw_noise`, `prepare_latents`, `encode_conditions`, `denoise`, and for images `plan`, `plan_denoise` and `step` | prefix cache units, latent pages, per-slot state rows |
| Image, video and audio decoders | `DecoderRunner` | `decode`, `quantize` | latent pages, products |

Host media on CPU workers has no model: the Rust `MediaRunner` calls the functions of `uniserve.media` directly. Video request rules (canvases, frame counts, how each condition is prepared and the conditioner's presentation) are implemented once, in `uniserve-core`: the head plans and presents every request with them before admission, and the worker derives the frame counts it admits from them. Python reaches the canvas rule, the training buckets, the canvas a target names and the frame count of a duration through functions the worker library exports (`rules.rs`); it does not plan or present requests.

A method that can be captured meets four conditions: its shapes are fixed by its scalar arguments; every value that changes between forwards enters as tensor contents; it performs no host planning or synchronization; it draws no random numbers, because draws and noise are inputs the control plane generates (for example a latent encoder's posterior noise, drawn by `draw_noise` from the request's seed).

### Library functions and functions Rust gives Python

Besides the runners, Rust calls:

- `uniserve.loading.rank`: read a checkpoint's configuration and the model's description, load this worker's modules;
- `uniserve.loading.host_weights`: the layer stack of a component that keeps its weights in host memory, its layout in a layer slot, and the rebinding of its projections to the slots Rust allocates;
- `uniserve.distributed`: process groups, created in the order Rust fixes; `collectives.NativeCollectives`, which wraps Rust's collectives for layers; `experts`, with `uniserve.backends.experts.HostBootstrap`, for expert unions;
- `uniserve.bindings.context.Scratch`: the transient work area that the execution contexts serialized on one stream share (a runner prepares its own execution context with `uniserve.runner.context.execution_context`);
- `uniserve.cuda_graph`: graph pools, `capture_begin`, `capture_end`, `break_graph`;
- `uniserve.prefix_cache`: the cache's page plan, unit resets, imported-page conversion;
- `uniserve.backends.kernel_choices`, `uniserve.profiling` and `uniserve.media`;
- `torch` and `gc` functions as they are.

In the other direction, Rust passes Python the collectives of a communicator on one stream, the host group's `all_gather`, `broadcast`, `barrier` and `split`, the peer-mapped storage Rust allocates for split expert weights (a storage tensor and every member's address), expert dispatch and weight prefetch, and the hooks that make a host-resident layer readable before its projections run and record their reads after.

### Attention providers

The library's provider registry (`uniserve.backends.attention.PROVIDERS`, with `resolve`) is the one list of attention providers: `auto` (a native kernel per call, chosen from the device and the installed kernel libraries), `flashinfer`, `trtllm`, `flash_attn_4`, `torch`, `prefix_block`, `flash_attn` and `sgl_kernel`. `attention_backend` passes its value to the workers unchanged, and a worker refuses a name the registry does not provide when it loads its model; the head reports that startup error. A provider declares whether serving captures its calls (`Backend.captures`); the portable `torch` provider does not. A host-planned provider's plan is shared by every layer of one execution context that has the same layer shape and batch signature, so a forward plans once.

## CUDA graphs

A captured forward is a CUDA graph, possibly breakable into segments separated by graph breaks: native work Rust recorded at capture and performs at launch, or one call on the Python thread.

- **Capture.** The control plane picks a runner method, a shape and the tensors for that shape, and records the capture into a job of the lane that serves the forward: the Python thread calls `capture_begin`, the method and `capture_end`, and the completion carries one raw executable per segment. Warmups are eager calls recorded the same way. Each runner captures each kind of forward at its configured shapes largest first, so smaller captures reuse the pool blocks of larger ones.
- **Launch.** For a forward with a graph the control thread submits one job. The execution thread writes the inputs with native copies, launches each segment with `cuGraphLaunch`, performs the breaks between segments and records an event; once it completes, it reads the outputs and posts the completion. A forward whose breaks are all native and whose attention reader is device-driven runs no Python; with a host-planned reader, the job first runs the runner's `plan_*` method on the Python thread.
- **Eager.** A forward without a graph writes the same inputs, hands the same method with the same arguments to the Python thread, then records the event and reads outputs as a launch does.
- **Shape key.** Each runner keeps its graphs in its own table, keyed by the runner's key (the method, the shape bucket, the numerical variant and the output selection) and, for an expert forward, the agreed capacity. A runner's tensors are fixed when it is constructed, so they are not part of the key.
- **Policy.** Rust decides which shapes are captured, in which order and pool. Once startup seals the graph tables, a configured bucket that a forward selects without a resident graph is an error, never an eager fallback; a text or diffusion LM forward that no configured bucket holds runs eagerly, so the captured buckets bound no batch the engine forms. The graph policy is resolved after loading: with an attention provider whose calls are not captured, `auto` serves every call eagerly and `full` is refused. The workers of a group capture the same shapes in the same order, since capture and launch include collectives. No host callback that enters Python is ever placed on a stream or inside a graph.

## Memory and storage

| Owner | Allocated by | Contents |
| --- | --- | --- |
| Control plane | the CUDA driver, from Rust | everything that holds request state, is read or written natively, or crosses a process: the prefix cache unit pool, slot tables, the latent pool, product arenas, forward inputs and outputs, pinned staging, export ranges |
| Resource plane | PyTorch's caching allocator | weights, activations, operator workspace, graph pools |

Capacity planning measures free device memory after loading and assigns the remainder to the prefix cache and media storage. The process's own usage, as NVML reports it, counts with the device storages its scope keeps at the host's Storage Server; exclusive storages a predecessor left that no owner has taken yet count as free, and an owner that allocates new memory first releases them. Storage is organized by owner in `crates/worker/src/storage`:

- **Prefix cache** (`prefix_cache/`): the unit pool, block tables, unit accesses, transfer bases and pending imports of attention state.
- **Latent cache** (`latent/`): denoising trajectories in two banks of pages per slot, with versions exported to other workers.
- **Product store** (`products/`): tensors one call produces and another consumes, with their reads, writes, exports and imports.
- **Request slots** (`request_slots.rs`): the per-slot state banks of the requests a runner holds.
- **Buffers** (`buffer.rs`): the `BufferPool` of cross-call buffers, bound at the offsets the scheduler placed them. Each worker backs its own pool, sized to the products of the components it executes and of the components that read them (`MediaCall::readers`), and reports its bytes; the engine places each media request's products in the pool of every worker that backs them, and a token runtime's products at one offset on every worker.
- **Outputs and host buffers**: page-locked readback of results (one arena per runtime on each lane, allocated when that runtime starts, of blocks the size of its largest batch; a batch leases exactly its words), and the `PinnedPool` of page-locked blocks that host-prepared inputs (decoded images) are written into.

### The Storage Server

Each host runs one `uniserve-storage-server` per serving user, which the head and each launcher start when their host has none, in a session of its own, so restarting the head or a launcher never ends it. Clients connect through a Unix socket owned by the user (`$XDG_RUNTIME_DIR/uniserve/storage.sock`, else `/tmp/uniserve-<uid>/storage.sock`) and pass file descriptors with `SCM_RIGHTS`. The server holds memory and never interprets it: it keeps the descriptors of CUDA VMM allocations (POSIX file descriptor handles) and host memfds, never creates a CUDA context, and moves no bytes; clients compute every key and copy on their own streams.

- **Claims.** The head claims its deployment (its served model name) at every host's server and hands each worker the claim's token in its launch descriptor. A claim releases the scopes the head no longer places; `uniserve-storage-server retire --deployment NAME` releases a deployment that is no longer served, and `list` prints what the server holds. In a hot update the successor head takes the predecessor's claim over unchanged instead of issuing a new one, so the workers that started under it keep their access (see Hot update).
- **Weights.** A GPU worker keeps its loaded parameters and buffers as read-only tensors, with a manifest of each tensor's checkpoint source and representation (`uniserve/loading/retained.py`). A tensor is keyed by its placement, its GPUs and the fingerprint of the checkpoint source it is loaded from (that source's file headers, the normalized configuration, the weight choices, and a digest of the installed `uniserve` and `uniserve_kernels` sources); the manifest and the tensors no source loads are keyed without the checkpoint's files. A starting worker maps the tensors it finds read-only and binds them instead of reading the checkpoint. Loading is collective, so the workers that load together agree on what to read: the tensors any of them misses, or everything when one has no manifest; a change to one checkpoint source thus reloads only its tensors. Each worker moves what it read into retained memory (one allocation per torch storage, small ones packed), publishes it and maps it read-only. The host region of a layer stack streamed from host memory is a read-only host storage among them. Loading stores every encoded weight in the layout its kernel provider declares (`uniserve/loading/layout.py`), and preparation verifies it, so the retained bytes are those the kernels read.
- **Exclusive storages.** The prefix cache planes, the buffer pool and the latent banks are exclusive storages with one writer, keyed by their scope, name and shape: a restarted worker of the same scope takes the memory its predecessor left, after the server shoots that predecessor down if it still holds write access (SIGKILL to its process group, its exit, and NVML no longer listing it on the GPU). An owner resets only memory created for it; a storage a predecessor left keeps the finite values it last wrote. A standby maps them while its predecessor serves from them, and a successor head of a hot update reads the cache pages its predecessor handed over; after a crash the engine's cache index starts empty, so nothing reads them before a unit is recycled for its next owner.
- **Host page tier.** The server keeps page pools of host memory, one memfd of equal slots per page shape and NUMA node, shared by every worker whose pages have that shape; a worker takes the pool on the host NUMA node closest to its GPU, whose pages the server binds to that node. With `kv_cache_config.host_tier`, each GPU worker maps the pool of its full-attention cache group; the scheduler pins the pages that follow a request's device prefix match and has its first prefill batch read them into new units, and writes every published prompt page into a reserved slot with a later batch, which its output commits (`crates/engine/src/prefix_cache/tiers.rs`). A page is one page of one group on one rank, layer-major: keys, values, and FP8 scales.
- **Distributed tier.** With `kv_cache_config.mooncake`, the server runs a Mooncake connector, a child process it starts again when it ends (`crates/storage/src/connector.rs`). The connector maps every page pool, registers it with Mooncake, and writes each committed page to Mooncake as the object `uniserve_{page key}_{kv head offset}-{kv head count}`, which names no deployment or host. A lookup counts the pages Mooncake holds past the pool's run. When Mooncake extends the run every rank's pool holds by at least 256 tokens, admission prefetches those pages into reserved slots, waits for them at most 1 s plus 0.25 s per 1024 tokens, and computes the pages that did not arrive. Prefetches hold at most half of a pool's slots. Mooncake's master evicts objects; the connector never removes them.

### Transfers

Transfers go through one module per mechanism in `crates/worker/src/transfer`: `local` (in-process borrows), `posix_shm` (POSIX shared memory on one host), `vmm` (CUDA VMM allocations exported in place or through a pool, imported by fabric handle or by a descriptor from the exporter's grant socket) and `channel` (bytes carried in the location). A read runs on a transfer-pool thread with one stream per thread and device: it claims the chunk and waits on the host for the claim, orders its copy behind the handoff and the producer's event (a cross-device read waits for the producer on the host), copies, acknowledges, and drains its stream before its task completes, which returns its bytes and read credit. A VMM read imports the allocation's handle, maps it and opens its ready event. An export states whether its allocation is resident (the producer's arenas, pools and cache planes, kept until it closes) or made for that export alone: a reader keeps its mapping of a resident fabric allocation for every later read of it until its transport closes, and releases any other mapping with the read. A worker's transfers share one byte budget and one set of read credits; an exported product holds its bytes once, whatever number of mechanisms publish it, and they return when its last backing physically retires. A worker holds one read credit per queued call (a media worker at least one per concurrent artifact import of each slot and one per worker of the deployment) and reports their count (`WorkerInfo::transfer_tickets`); the engine keeps no more transfer calls reserved than the narrowest worker holds.

## Startup

Every worker of a group runs these phases in the same order, because several are collective:

1. The control thread connects to the head, registers and receives the launch descriptor while the Python thread starts the interpreter; until serving starts, a head watch ends the worker at once if the head's connection closes. Once the interpreter is up, Rust checks the devices and makes their primary contexts current.
2. `read_config` describes the model; Rust validates placement.
3. Rust plans the process groups; Python creates them.
4. `load_model`, or binding the weights the worker retained when it last loaded the same model on the same placement; the checkpoint reader reads each source file front to back into the page cache before mapping its tensors, so workers loading one cold file do not fault it page by page. A worker that loaded moves its weights into retained memory and publishes them.
5. Rust creates the worker's streams and starts one execution thread per lane; expert unions and weight prefetch start, and a component placed with `"weights": "host"` moves its layer projections into a page-locked host region and a ring of device slots that Rust copies them into ahead of each layer.
6. Rust plans capacity from the model's sizes and measured memory, allocates request storage, and constructs a runner, with its execution context, for each placed component and lane.
7. The executor and the profiler start.
8. Warmup: Rust sets each runner's graph budget and prepares it; then every captured method runs eagerly once at its shape, so kernels compile outside capture, and is captured, each runner's shapes largest first. Paths that always run eagerly (image prompts, uncaptured image denoising steps) run once too, so no serving request selects a kernel first.
9. The runners' graph tables and graph storage are sealed; Python's garbage collector is collected, frozen and disabled; the worker reports its capacities and the shapes it prepared to the engine: one image shape per captured guidance-branch count (`WorkerInfo::image_warmups`, a guided one with its model's framing of an empty negative prompt) the rows of one diffusion LM sampler chunk (`diffusion_lm_step_rows`), and the tokens its image denoiser's image-unconditional guidance branch attends to when that branch does not read the request's own context (`image_unconditional_token_ids`), which the engine reserves and prefills as a guidance prefix of its own beside the negative prompt's.
10. On GPU workers the engine warms the serving path with real requests through its scheduler (a successor head of a hot update, whose workers park until it takes over, runs none) (`crates/engine/src/warmup.rs`): a token request of a prefill and a decode, a group of equal one-block diffusion LM requests for each row count of a sampler chunk, each submitted in one command so its steps run with exactly that many rows (the one-row request generates a second block, so a commit prefill runs), and a two-step image request per image shape at the guidance scales of its branch count. A request the deployment's generation limits refuse a client is left out. Serving starts once every warmup request finished.

After startup the Python thread is idle except for eager forwards, host-planned attention plans and the little resource work issued while serving.

## Serving

**A text decode step.** The engine sends a `Batch` whose decode calls name request slots. `CausalLMRunner` picks the decode graph of the smallest row bucket that holds the batch and submits the job. The execution thread writes the forward's inputs (slot indices, finish flags, the rows' sampling parameters and draws) and launches: the graph gathers tokens, positions and attention metadata from slot state on the device, runs the model, projects the final-position logits and selects each row's token: greedily, or by sampling under the row's temperature, top-k (up to 128 candidates), top-p, min-p and penalties with the draws Rust generated from the request's random stream, all of them tensor contents of the graph. A batch that needs token sets, logit biases, typical-p, draft tokens, reported logprobs or a sampled top-k beyond the 128 candidates selects eagerly on the Python thread with the same kept-set computation; reported logprobs are scored on the model's raw distribution. The commit graph installs the selections into slot state, so the next step can be queued before this one's result is read. When the forward's event completes the execution thread reads the selection words and posts the completion; the control thread updates request state and returns the `BatchOutput`. Prefill (which also carries verification rows) and commit have their own graphs, and a verification selects eagerly with `verify`; with tensor parallelism every worker launches the same graph and the collectives inside it were captured with it.

**A FastH3 video request.** The text encoder's graph of the smallest prompt capacity that holds the prompt runs on its tensor-parallel group; `prepare_latents` writes the request's noise and state rows; each denoising step gathers the slot's committed latent pages into the sample tensors, launches the layout's graph on every worker of the sequence-parallel group and scatters the result into the other bank; video and audio decoding launch one graph per temporal unit; the decoded units transfer to the CPU workers, which encode each unit and assemble the MP4.

**Batches.** Every capability's batch goes through the same steps (`crates/worker/src/executor/batch.rs`): open (bind calls to their requests, read host-gated predicates), lease output storage, bind completion flags, seal (end reads, validate every owner's update), commit once the job carrying the device work completed (writes become visible, calls pending), and materialize once the output fences complete, finishing any host work such as image or video encoding. A failure before the commit point fails the batch's calls and the worker keeps serving; a failure at or past it is fatal to the worker.

## Parallelism

- **Lanes.** A lane is a stream with its execution thread, optionally on its own SM partition; lanes overlap on the device and do not wait for each other on the host.
- **Collectives.** A layer calls its communicator; in serving its functions are Rust's NCCL collectives on the forward's stream, captured with the forward.
- **Expert forwards.** The workers of an expert union agree on each forward's source and capacity over their host group, then launch; a worker without tokens launches the join graph of the agreed capacity. Expert dispatch with host work, weight prefetch and microbatch boundaries are graph breaks.
- **Engine cores.** Each data-parallel copy of scheduling is an engine core: its own scheduler thread and the worker groups it drives. A replica is one of the groups a deployment file's `replicas = N` expands a group into.

## Failure handling

Errors cross with a stable code, whether they are fatal to the worker, and the phase they occurred in. A Python exception in an eager forward fails that forward's batch. An error in resource work fails startup, or at serving time the batch that needed the resource. A launch failure or a driver error is fatal to the worker, and the engine replaces its failure domain: the worker group, or every group that shares an expert world, whose workers hold that world's communicators. Every group of the domain terminates its workers and fails the requests they held; the domain is relaunched together, its groups form their communicators again, and their weights and exclusive memory come back from the host's Storage Server. A group that keeps a standby (`worker_config.standby`) is replaced by promoting it instead: its ranks started beside the group's, mapped the group's weights and exclusive storages without write access, captured their graphs on the warmup scratch units, which the scheduler never allocates, and parked; on promotion each takes write access once the Storage Server has the failed rank out and serves at once, and the group starts a new standby (`crates/engine/src/worker/group/standby.rs`).

A driver call that does not return stops only its lane's execution thread. The control thread keeps serving the channel and, once a job has stayed in flight for the stall bound (half the head's watchdog timeout, `worker_config.watchdog_timeout`, sent in the launch descriptor), reports to the head the lane (or the Python thread), the batch, its runner method, its calls and how long it has run (`WorkerResponse::Stalled`). The head logs the report, fails the affected requests with an error naming the lane, and replaces the group. The head's watchdog covers a worker that stops answering altogether.

A worker's orderly close ends with a reset of its primary contexts before the process exits. The reset releases the driver state the process created, including the shared-memory file (`/dev/shm/cuda.shm.*`) the CUDA driver makes for a process's interprocess events and removes only when the context is destroyed. A worker that loses its head with nothing in flight takes the aborted close and then the reset; with work in flight it ends at once. A worker that ends without its orderly close first removes the driver file and the NCCL shared-memory names (`/dev/shm/nccl-*`) its mappings still link. The process that reaps a worker (the head for its local workers, `uniserve-launcher` for remote ones) observes the death before reaping and removes that worker's driver files and recorded NCCL names while the zombie still holds its pid. A worker's startup failure, the interpreter's included, reaches the head as the worker's own error.

## Hot update

A deployment moves to a new build of the head and the worker runtime without failing a request or closing a client connection. Running `uniserve serve` again with the same configuration on the head's host starts the successor: every head offers its deployment on a handover socket in the Storage Server's directory, named after the deployment, and a head that finds one offered succeeds the head that offers it (`crates/server/src/handover.rs`).

1. The successor states its protocol and snapshot versions, its deployment and its engine core count; the predecessor refuses other ones. The predecessor's groups terminate their standbys, whose memory the successor's workers take, and the predecessor answers with its storage claim, each group's cache units and a duplicate of its listening socket.
2. The successor launches its workers standing by for the predecessor's: under the predecessor's claim, they map the retained weights and the exclusive storages of the same cache units without write access, warm up and capture their graphs on the warmup scratch units, and park. It then asks for the handover.
3. Each of the predecessor's engine cores quiesces (`crates/engine/src/scheduler/handover.rs`): it admits no further token request, runs the requests it cannot migrate to completion, pauses the rest, and lets every batch in flight drain. At that step boundary it publishes each running request's computed cache pages (its prompt and its generated tokens) to the prefix cache, snapshots the requests and the prefix cache's index, and closes its workers, which release their exclusive storages with the contents intact.
4. The predecessor sends the snapshots and stops accepting connections; the successor accepts on the same socket from then on.
5. The successor takes the claim over and promotes its workers, which take write access in place. Each engine core installs the predecessor's pages under its own workers' endpoints and queues every migrated request with its generated tokens after its prompt, under its request identity and epoch, so its sampling stream continues and its penalties count those tokens (`GenerationParams::generated_token_ids`). Admission finds the request's pages in the prefix cache, and its first prefill computes at most the last partial page and samples its next token.
6. The predecessor keeps serving the connections it holds: their clients' commands (cancellation, stop-string decisions) and the token requests they submit go to the successor, and the events of those requests come back. It exits once its last connection closes.

A request migrates when it is a token request that waits, or that runs with a text prompt, generates text only, has published any prompt scores it asked for and has not finished. Media requests, token requests with images, readouts and canvases complete on the predecessor before it quiesces, which lengthens the pause by their remaining time, as does a finished request whose last tokens await its client's stop-string decisions. A request whose cache policy writes no pages recomputes its context on the successor. A media request a client of the predecessor submits after the handover is refused as overloaded and its retry reaches the successor. Asynchronous video jobs stay with the predecessor.

The handover serves deployments whose groups all run on the head's host and share no expert world, as standbys do; any other deployment is refused. A successor that leaves before it asks for the handover leaves the predecessor serving, with its standbys started again; one that fails afterwards fails the migrated requests.

## Repository layout

```text
crates/core/        uniserve-core: shared domain values (ids, sampling, generation, media, parallel placement) and the video request rules
crates/ipc/         uniserve-ipc: head-to-worker messages, the channel, registration, launch descriptor, launcher messages
crates/cuda/        uniserve-cuda: CUDA driver, VMM, NVML and NCCL bindings and graph launch, loaded at run time
crates/engine/      uniserve-engine: scheduler, prefix cache allocation, worker groups and process launch
crates/server/      uniserve-server: HTTP/OpenAI API, model profiles, request lowering and assembly
crates/worker/      uniserve-worker: the worker runtime (installed as the library uniserve/_worker) and the uniserve-worker executable
crates/observability/         metrics registry
crates/observability-derive/  the registry's derive macro
crates/bin/         uniserve (head), uniserve-launcher (launcher), uniserve-dynamo (Dynamo head)
uniserve/           the numerical library, the concrete models, the Python half of each runner and the worker's Python entry
uniserve_kernels/   Triton, CuTe and CUDA kernels, and the vendored DeepEP and DeepGEMM sources
uniserve_eval/      HTTP evaluation client
```

Dependency direction: `core <- ipc <- {engine, worker, uniserve-launcher}`, `cuda <- worker`, `{core, engine, observability} <- server <- {uniserve, uniserve-dynamo}`. The worker never depends on the engine or server, and the engine never links CUDA.

| Worker runtime module (`crates/worker/src`) | Responsibility |
| --- | --- |
| `launch`, `config` | the worker entry, arguments, signals, the Python thread's interpreter, registration with the head; the worker's configuration; component declarations and placement |
| `worker` | start, serve, ordered close; the model description and loading, capacity planning (`worker/planning.rs`) and request storage allocation (`worker/storage.rs`) |
| `warmup` | the startup phases after storage: graph budgets, preparation, capture, completion (`WarmupHooks`); the serving path is warmed by the engine |
| `service.rs` | serves the channel |
| `completion`, `error`, `logging` | the completions jobs post; error kinds; tracing setup |
| `executor` | batch admission, forward order, job submission, completions, retirement; the batch skeleton and the image and video batches |
| `runner` | the Rust half of each runner, attention metadata, input buffers, `MediaRunner` |
| `lane`, `stream` | lanes, jobs, job queues and completions; the worker's streams and the event pool |
| `ffi` | `Interpreter`, `Function`, `Handle`, DLPack export of Rust storage |
| `cuda_graph` | shape keys, graph tables, graph pools |
| `request`, `sampling` | the request pool and progress; a forward's sampling inputs |
| `storage`, `transfer` | as described above |
| `parallel` | host groups, process-group topology, expert parallelism |
| `host`, `capacity`, `profiling` | host lanes, capacity planning, NVTX ranges, kernel records and profiler windows |
| `rules.rs` | the video canvas and frame-count rules exported to Python |

| Python package (`uniserve`) | Content |
| --- | --- |
| `worker.py`, `errors.py` | `attach`, `import_path` and the library's loader; the error kinds a failure crosses with |
| `profiling.py` | NVTX ranges and profiler windows Rust drives |
| `processing.py`, `rotary.py`, `tensors.py` | caller-owned image transforms and diffusion prompt framing; checkpoint rotary metadata normalized into typed recipes; numerical buffer requirements (`BufferConfig`) and borrowed tensor outputs |
| `runner` | the Python half of each runner, attention metadata, the execution context a runner prepares |
| `models` | capability classes, model inputs, component declarations, the checkpoint registry and the concrete models (`qwen3`, `qwen3_vl`, `siglip`, `bagel`, `sensenova_u1`, `minimax_h3`, `diffusion_gemma`, `stub`) |
| `loading` | checkpoint resolution, configuration normalization, weight mapping and a worker's model loading (`loading.rank`) |
| `nn`, `backends`, `bindings`, `cuda_graph` | layers, kernel providers, prepared operators and execution contexts, breakable CUDA graphs |
| `prefix_cache`, `diffusion`, `sampling`, `quantization`, `distributed`, `media` | prefix cache numerics, schedules and solvers, sampling, quantized tensors, partitioning and the process world, codecs |
