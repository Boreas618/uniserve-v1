# Architecture

UniServe is a Rust serving system with a Python numerical library. Rust owns every host responsibility: HTTP, scheduling, the rank processes and the channel to them, request state, storage, CUDA streams and graph policy, transfers, collective orchestration, media containers and observability. The Python package `uniserve` owns numerical computation: model definitions, layers, kernel bindings, samplers and solvers. The two languages meet only in function calls through TVM-FFI.

This document describes how a deployment is organized, how a rank process divides its work between Rust and Python, and how a request travels through it.

## Principles

1. **Rust is the control plane.** Every decision and every piece of request state is Rust's: requests, slots, progress, sampling policy, random streams, storage assignment, graph policy, stream ordering, transfers and failure handling.
2. **Python is stateless.** Python provides resources and numerical execution on demand and keeps nothing about a request between calls.
3. **The languages meet only in function calls.** A call passes tensors, scalars, flat lists of integers or floats, strings and opaque handles, and returns the same. Neither side depends on a memory layout, record or encoding that the other defines.
4. **Each behavior is implemented once.** A forward's computation is one method of a runner, used by serving, by graph capture and by Python callers alike. A rule both languages need is written in one of them and called from the other.
5. **Python never runs concurrently with itself.** In a rank process at most one thread executes Python at any instant, so no thread waits for the GIL.

## Processes

| Process | Language | Role |
| --- | --- | --- |
| `uniserve` (head) | Rust | HTTP API, model profile, tokenization, request lowering and response assembly (`uniserve-server`); scheduling, storage assignment and the rank fleet (`uniserve-engine`) |
| `uniserve-host` | Rust | Starts and supervises the ranks placed on a host other than the head's |
| `dynamo-worker` | Rust | An experimental head for FastH3 that embeds the engine and the serving runtime in a Dynamo worker; Dynamo's frontend replaces the HTTP API |
| Rank | Rust runtime hosted by a Python interpreter | One per device rank and per CPU host rank; executes the batches the engine sends |

The head contains no Python. A rank starts as `python -m uniserve.worker`, which loads the rank runtime library `uniserve._worker` (a TVM-FFI library, not a Python extension module) with `tvm_ffi.load_module` and calls its exported `main(argv, import)`; `import` is `uniserve.worker.import_path`, the one function through which Rust reaches the library. From then on Rust runs the process, and the process exits with the status `main` returns.

### The channel

The engine and each rank exchange the messages of `uniserve-ipc` (`NewRequest`, `Batch`, `BatchOutput`, `WorkerInfo`) over one stream connection per rank.

- **Connection.** The head listens on one Unix socket path and one TCP port. A rank connects to the head when it starts: through the Unix socket on the head's host, through TCP from any other host. Ranks and launchers accept no channel connection; the connections ranks accept are their host group's peer mesh and its collective store.
- **Registration.** A rank's first frame registers it: worker, rank, launch generation and build. The head numbers each launch of a worker group, and a relaunch takes the next number; the build is the package version plus a digest of the sources every artifact is compiled from, so the head, launchers and ranks of one build carry the same value. The head refuses a registration of another build or of a launch that is not the group's current one, and otherwise answers with the rank's launch descriptor, which carries every other launch value.
- **Lifetime.** The same connection then carries requests and responses, and its close is the only death signal of a registered rank: the head treats the rank as dead and the rank exits. Before a rank registers, the head notices the exit of a process it started itself, and a launcher reports the exits of the ranks it started.
- **Frames.** A frame is a length followed by one message in a serde binary encoding (MessagePack). Head and ranks are built from one workspace, so the message types are the protocol's only definition. A frame carries a message of any size its `u32` length states; no launch setting bounds it further.
- **Waiting.** Each side blocks on its socket together with one local wake descriptor, which its own threads fire for work that did not arrive on the socket.
- **Launchers.** A `uniserve-host` launcher's connection carries the same frames: its registration, the head's instructions to reserve, start and stop a group's ranks and to terminate, and its reports of reservations and rank exits.
- **Data.** Tensors, prefix cache pages, latents and products move through the transfer mechanisms. Only a host product routed between hosts travels on the channel.

## The planes of a rank

| Plane | Thread | Owner | Blocks only on | Does |
| --- | --- | --- | --- | --- |
| Control | `control` | `Service`, `Executor`, the Rust half of each runner | Its socket and wake descriptor | Serves the channel, holds request state, admits batches, prepares each forward's inputs, picks its graph, builds its job, applies completions, retires storage |
| Execution | one per lane (`lane-N`) | `Lane`, `Job` | The work on its own stream | Owns the lane's stream: input copies, graph launches, event records and waits, output reads; hands Python work for its stream to the Python thread and waits for it |
| Resource | the Python thread | `Interpreter` | The interpreter | Executes every Python call, one at a time: loading, execution contexts, captures, eager forwards, releases |

Threads exchange messages only. The control thread submits jobs and receives completions; it calls neither the driver nor Python while it serves, so a slow launch, a driver call that waits on the device, or a stopped lane does not keep it from serving the channel, handling cancellation or reporting status. A lane's execution thread is the only issuer of work on its stream, so the stream's order is that thread's order. A lane posts a job's completion once it has issued the job's work, not once the device has finished it: memory a job used is read, rewritten or freed afterwards only behind something that waited for the device (a read in the job, a stream wait the job recorded, events the job completes after, or the fences that retire storage). Every Python call runs on the Python thread, in arrival order: an execution thread hands it a call and waits for the result; the control thread submits work that owns what it uses and never waits for it; a released Python object is dropped there in queue order, and nobody waits for the drop.

Eager expert forwards with several microbatches are the one exception to a single thread per lane: each forward stays suspended inside Python at its expert dispatch points, so each microbatch has a thread of its own, and those threads call Python themselves as cooperative turns, exactly one at a time, so they never contend for the GIL.

Blocking host work runs on host lanes, threads (`<lane>-<index>`) that each serve one kind of work and hand any Python call to the Python thread like every other thread: media (`uniserve-media`), image preparation (`uniserve-image`), prefix cache imports (`worker-cache-import`), transfer reads (`uniserve-<kind>-read`) and shared-memory retirement (`uniserve-posix-shm-retire`). Signals (`worker-signals`), VMM descriptor grants (`worker-descriptor-grants`) and, during startup, the head watch (`uniserve-head-watch`) are dedicated threads.

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

Host media on CPU ranks has no model: the Rust `MediaRunner` calls the functions of `uniserve.media` directly. Video request rules (canvases, frame counts, how each condition is prepared and the conditioner's presentation) are implemented once, in `uniserve-core`: the head plans and presents every request with them before admission, and the rank derives the frame counts it admits from them. Python reaches the canvas rule, the training buckets, the canvas a target names and the frame count of a duration through functions the rank runtime library exports (`rules.rs`); it does not plan or present requests.

A method that can be captured meets four conditions: its shapes are fixed by its scalar arguments; every value that changes between forwards enters as tensor contents; it performs no host planning or synchronization; it draws no random numbers, because draws and noise are inputs the control plane generates (for example a latent encoder's posterior noise, drawn by `draw_noise` from the request's seed).

### Library functions and functions Rust gives Python

Besides the runners, Rust calls:

- `uniserve.loading.rank`: read a checkpoint's configuration and the model's description, load this rank's modules;
- `uniserve.loading.host_weights`: the layer stack of a component that keeps its weights in host memory, its layout in a layer slot, and the rebinding of its projections to the slots Rust allocates;
- `uniserve.distributed`: process groups, created in the order Rust fixes; `collectives.NativeCollectives`, which wraps Rust's collectives for layers; `experts`, with `uniserve.backends.experts.HostBootstrap`, for expert unions;
- `uniserve.bindings.context.Scratch`: the transient work area that the execution contexts serialized on one stream share (a runner prepares its own execution context with `uniserve.runner.context.execution_context`);
- `uniserve.cuda_graph`: graph pools, `capture_begin`, `capture_end`, `break_graph`;
- `uniserve.prefix_cache`: the cache's page plan, unit resets, imported-page conversion;
- `uniserve.backends.kernel_choices`, `uniserve.profiling` and `uniserve.media`;
- `torch` and `gc` functions as they are.

In the other direction, Rust passes Python the collectives of a communicator on one stream, the host group's `all_gather`, `broadcast`, `barrier` and `split`, the peer-mapped storage Rust allocates for split expert weights (a storage tensor and every member's address), expert dispatch and weight prefetch, and the hooks that make a host-resident layer readable before its projections run and record their reads after.

### Attention providers

The library's provider registry (`uniserve.backends.attention.PROVIDERS`, with `resolve`) is the one list of attention providers: `auto` (a native kernel per call, chosen from the device and the installed kernel libraries), `flashinfer`, `trtllm`, `flash_attn_4`, `torch`, `prefix_block`, `flash_attn` and `sgl_kernel`. `attention_backend` passes its value to the ranks unchanged, and a rank refuses a name the registry does not provide when it loads its model; the head reports that startup error. A provider declares whether serving captures its calls (`Backend.captures`); the portable `torch` provider does not. A host-planned provider's plan is shared by every layer of one execution context that has the same layer shape and batch signature, so a forward plans once.

## CUDA graphs

A captured forward is a CUDA graph, possibly breakable into segments separated by graph breaks: native work Rust recorded at capture and performs at launch, or one call on the Python thread.

- **Capture.** The control plane picks a runner method, a shape and the tensors for that shape, and records the capture into a job of the lane that serves the forward: the Python thread calls `capture_begin`, the method and `capture_end`, and the completion carries one raw executable per segment. Warmups are eager calls recorded the same way. Each runner captures each kind of forward at its configured shapes largest first, so smaller captures reuse the pool blocks of larger ones.
- **Launch.** For a forward with a graph the control thread submits one job. The execution thread writes the inputs with native copies, launches each segment with `cuGraphLaunch`, performs the breaks between segments and records an event; once it completes, it reads the outputs and posts the completion. A forward whose breaks are all native and whose attention reader is device-driven runs no Python; with a host-planned reader, the job first runs the runner's `plan_*` method on the Python thread.
- **Eager.** A forward without a graph writes the same inputs, hands the same method with the same arguments to the Python thread, then records the event and reads outputs as a launch does.
- **Shape key.** Each runner keeps its graphs in its own table, keyed by the runner's key (the method, the shape bucket, the numerical variant and the output selection) and, for an expert forward, the agreed capacity. A runner's tensors are fixed when it is constructed, so they are not part of the key.
- **Policy.** Rust decides which shapes are captured, in which order and pool. Once startup seals the graph tables, a configured bucket that a forward selects without a resident graph is an error, never an eager fallback; a text or diffusion LM forward that no configured bucket holds runs eagerly, so the captured buckets bound no batch the engine forms. The graph policy is resolved after loading: with an attention provider whose calls are not captured, `auto` serves every call eagerly and `full` is refused. The ranks of a group capture the same shapes in the same order, since capture and launch include collectives. No host callback that enters Python is ever placed on a stream or inside a graph.

## Memory and storage

| Owner | Allocated by | Contents |
| --- | --- | --- |
| Control plane | the CUDA driver, from Rust | everything that holds request state, is read or written natively, or crosses a process: the prefix cache unit pool, slot tables, the latent pool, product arenas, forward inputs and outputs, pinned staging, export ranges |
| Resource plane | PyTorch's caching allocator | weights, activations, operator workspace, graph pools |

Capacity planning measures free device memory after loading and assigns the remainder to the prefix cache and media storage. Storage is organized by owner in `crates/worker/src/storage`:

- **Prefix cache** (`prefix_cache/`): the unit pool, block tables, unit accesses, transfer bases and pending imports of attention state.
- **Latent cache** (`latent/`): denoising trajectories in two banks of pages per slot, with versions exported to other ranks.
- **Product store** (`products/`): tensors one call produces and another consumes, with their reads, writes, exports and imports.
- **Request slots** (`request_slots.rs`): the per-slot state banks of the requests a runner holds.
- **Buffers** (`buffer.rs`): the `BufferPool` of cross-call buffers, bound at the offsets the scheduler placed them. Each rank backs its own pool, sized to the products of the components it executes and of the components that read them (`MediaCall::readers`), and reports its bytes; the engine places each media request's products in the pool of every rank that backs them, and a token runtime's products at one offset on every rank.
- **Outputs and host buffers**: page-locked readback of results (one arena per runtime on each lane, allocated when that runtime starts, of blocks the size of its largest batch; a batch leases exactly its words), and the `PinnedPool` of page-locked blocks that host-prepared inputs (decoded images) are written into.

Transfers go through one module per mechanism in `crates/worker/src/transfer`: `local` (in-process borrows), `posix_shm` (POSIX shared memory on one host), `vmm` (CUDA VMM allocations exported in place or through a pool, imported by fabric handle or by a descriptor from the exporter's grant socket) and `channel` (bytes carried in the location). A read runs on a transfer-pool thread with one stream per thread and device: it claims the chunk and waits on the host for the claim, orders its copy behind the handoff and the producer's event (a cross-device read waits for the producer on the host), copies, acknowledges, and drains its stream before its task completes, which returns its bytes and read credit. A VMM read imports the allocation's handle, maps it and opens its ready event. An export states whether its allocation is resident (the producer's arenas, pools and cache planes, kept until it closes) or made for that export alone: a reader keeps its mapping of a resident fabric allocation for every later read of it until its transport closes, and releases any other mapping with the read. A rank's transfers share one byte budget and one set of read credits; an exported product holds its bytes once, whatever number of mechanisms publish it, and they return when its last backing physically retires. A rank holds one read credit per queued call (a media rank at least one per concurrent artifact import of each slot and one per rank of the deployment) and reports their count (`WorkerInfo::transfer_tickets`); the engine keeps no more transfer calls reserved than the narrowest rank holds.

## Startup

Every rank of a group runs these phases in the same order, because several are collective:

1. The control thread connects to the head, registers and receives the launch descriptor; until serving starts, a head watch ends the rank at once if the head's connection closes. Rust checks the devices and makes their primary contexts current.
2. `read_config` describes the model; Rust validates placement.
3. Rust plans the process groups; Python creates them.
4. `load_model`; the checkpoint reader reads each source file front to back into the page cache before mapping its tensors, so ranks loading one cold file do not fault it page by page.
5. Rust creates the rank's streams and starts one execution thread per lane; expert unions and weight prefetch start, and a component placed with `"weights": "host"` moves its layer projections into a page-locked host region and a ring of device slots that Rust copies them into ahead of each layer.
6. Rust plans capacity from the model's sizes and measured memory, allocates request storage, and constructs a runner, with its execution context, for each placed component and lane.
7. The executor and the profiler start.
8. Warmup: Rust sets each runner's graph budget and prepares it; then every captured method runs eagerly once at its shape, so kernels compile outside capture, and is captured, each runner's shapes largest first. Paths that always run eagerly (image prompts, uncaptured image denoising steps) run once too, so no serving request selects a kernel first.
9. The runners' graph tables and graph storage are sealed; Python's garbage collector is collected, frozen and disabled; the rank reports its capacities and the shapes it prepared to the engine: one image shape per captured guidance-branch count (`WorkerInfo::image_warmups`, a guided one with its model's framing of an empty negative prompt) the rows of one diffusion LM sampler chunk (`diffusion_lm_step_rows`), and the tokens its image denoiser's image-unconditional guidance branch attends to when that branch does not read the request's own context (`image_unconditional_token_ids`), which the engine reserves and prefills as a guidance prefix of its own beside the negative prompt's.
10. On CUDA ranks the engine warms the serving path with real requests through its scheduler (`crates/engine/src/warmup.rs`): a token request of a prefill and a decode, a group of equal one-block diffusion LM requests for each row count of a sampler chunk, each submitted in one command so its steps run with exactly that many rows (the one-row request generates a second block, so a commit prefill runs), and a two-step image request per image shape at the guidance scales of its branch count. A request the deployment's generation limits refuse a client is left out. Serving starts once every warmup request finished.

After startup the Python thread is idle except for eager forwards, host-planned attention plans and the little resource work issued while serving.

## Serving

**A text decode step.** The engine sends a `Batch` whose decode calls name request slots. `CausalLMRunner` picks the decode graph of the smallest row bucket that holds the batch and submits the job. The execution thread writes the forward's inputs (slot indices, finish flags, the rows' sampling parameters and draws) and launches: the graph gathers tokens, positions and attention metadata from slot state on the device, runs the model, projects the final-position logits and selects each row's token: greedily, or by sampling under the row's temperature, top-k (up to 128 candidates), top-p, min-p and penalties with the draws Rust generated from the request's random stream, all of them tensor contents of the graph. A batch that needs token sets, logit biases, typical-p, draft tokens, reported logprobs or a sampled top-k beyond the 128 candidates selects eagerly on the Python thread with the same kept-set computation; reported logprobs are scored on the model's raw distribution. The commit graph installs the selections into slot state, so the next step can be queued before this one's result is read. When the forward's event completes the execution thread reads the selection words and posts the completion; the control thread updates request state and returns the `BatchOutput`. Prefill (which also carries verification rows) and commit have their own graphs, and a verification selects eagerly with `verify`; with tensor parallelism every rank launches the same graph and the collectives inside it were captured with it.

**A FastH3 video request.** The text encoder's graph of the smallest prompt capacity that holds the prompt runs on its tensor-parallel group; `prepare_latents` writes the request's noise and state rows; each denoising step gathers the slot's committed latent pages into the sample tensors, launches the layout's graph on every rank of the sequence-parallel group and scatters the result into the other bank; video and audio decoding launch one graph per temporal unit; the decoded units transfer to the CPU host ranks, which encode each unit and assemble the MP4.

**Batches.** Every capability's batch goes through the same steps (`crates/worker/src/executor/batch.rs`): open (bind calls to their requests, read host-gated predicates), lease output storage, bind completion flags, seal (end reads, validate every owner's update), commit once the job carrying the device work completed (writes become visible, calls pending), and materialize once the output fences complete, finishing any host work such as image or video encoding. A failure before the commit point fails the batch's calls and the rank keeps serving; a failure at or past it is fatal to the rank.

## Parallelism

- **Lanes.** A lane is a stream with its execution thread, optionally on its own SM partition; lanes overlap on the device and do not wait for each other on the host.
- **Collectives.** A layer calls its communicator; in serving its functions are Rust's NCCL collectives on the forward's stream, captured with the forward.
- **Expert forwards.** The ranks of an expert union agree on each forward's source and capacity over the host group, then launch; a rank without tokens launches the join graph of the agreed capacity. Expert dispatch with host work, weight prefetch and microbatch boundaries are graph breaks.
- **Replicas.** Each data-parallel replica is its own engine core, scheduler and rank group.

## Failure handling

Errors cross with a stable code, whether they are fatal to the rank, and the phase they occurred in. A Python exception in an eager forward fails that forward's batch. An error in resource work fails startup, or at serving time the batch that needed the resource. A launch failure or a driver error is fatal to the rank, and the engine replaces the rank group; a replica of an expert-parallel world is not relaunched, because its peers hold that world's communicators.

A driver call that does not return stops only its lane's execution thread. The control thread keeps serving the channel and, once a job has stayed in flight for the stall bound (half the head's watchdog timeout, `worker_config.watchdog_timeout`, sent in the launch descriptor), reports to the head the lane (or the Python thread), the batch, its runner method, its calls and how long it has run (`WorkerResponse::Stalled`). The head logs the report, fails the affected requests with an error naming the lane, and replaces the group. The head's watchdog covers a rank that stops answering altogether.

A rank's orderly close ends with a reset of its primary contexts before the process exits. The reset releases the driver state the process created, including the shared-memory file (`/dev/shm/cuda.shm.*`) the CUDA driver makes for a process's interprocess events and removes only when the context is destroyed. A rank that loses its head with nothing in flight takes the aborted close and then the reset; with work in flight it ends at once. A rank that ends without its orderly close first removes the driver file and the NCCL shared-memory names (`/dev/shm/nccl-*`) its mappings still link. The process that reaps a rank (the head for its local ranks, `uniserve-host` for remote ones) observes the death before reaping and removes that rank's driver files and recorded NCCL names while the zombie still holds its pid. A rank's startup failure reaches the head as the rank's own error.

## Repository layout

```text
crates/core/        uniserve-core: shared domain values (ids, sampling, generation, media, parallel placement) and the video request rules
crates/ipc/         uniserve-ipc: head-to-rank messages, the channel, registration, launch descriptor, launcher messages
crates/cuda/        uniserve-cuda: CUDA driver, VMM, NVML and NCCL bindings and graph launch, loaded at run time
crates/engine/      uniserve-engine: scheduler, prefix cache allocation, rank fleet and process launch
crates/server/      uniserve-server: HTTP/OpenAI API, model profiles, request lowering and assembly
crates/worker/      uniserve-worker: the rank runtime (installed as the module uniserve._worker)
crates/observability/         metrics registry
crates/observability-derive/  the registry's derive macro
crates/bin/         uniserve (head), uniserve-host (launcher), dynamo-worker
uniserve/           the numerical library, the concrete models, the Python half of each runner and the rank entry
uniserve_kernels/   Triton, CuTe and CUDA kernels, and the vendored DeepEP and DeepGEMM sources
uniserve_eval/      HTTP evaluation client
```

Dependency direction: `core <- ipc <- {engine, worker, uniserve-host}`, `cuda <- worker`, `{core, engine, observability} <- server <- {uniserve, dynamo-worker}`. The worker never depends on the engine or server, and the engine never links CUDA.

| Rank runtime module (`crates/worker/src`) | Responsibility |
| --- | --- |
| `launch`, `config` | arguments, signals, registration with the head; the rank's configuration; component declarations and placement |
| `worker` | start, serve, ordered close; the model description and loading, capacity planning (`worker/planning.rs`) and request storage allocation (`worker/storage.rs`) |
| `warmup` | the startup phases after storage: graph budgets, preparation, capture, completion (`WarmupHooks`); the serving path is warmed by the engine |
| `service.rs` | serves the channel |
| `completion`, `error`, `logging` | the completions jobs post; error kinds; tracing setup |
| `executor` | batch admission, forward order, job submission, completions, retirement; the batch skeleton and the image and video batches |
| `runner` | the Rust half of each runner, attention metadata, input buffers, `MediaRunner` |
| `lane`, `stream` | lanes, jobs, job queues and completions; the rank's streams and the event pool |
| `ffi` | `Interpreter`, `Function`, `Handle`, DLPack export of Rust storage |
| `cuda_graph` | shape keys, graph tables, graph pools |
| `request`, `sampling` | the request pool and progress; a forward's sampling inputs |
| `storage`, `transfer` | as described above |
| `parallel` | host groups, process-group topology, expert parallelism |
| `host`, `capacity`, `profiling` | host lanes, capacity planning, NVTX ranges, kernel records and profiler windows |
| `rules.rs` | the video canvas and frame-count rules exported to Python |

| Python package (`uniserve`) | Content |
| --- | --- |
| `worker.py`, `errors.py` | rank entry and `import_path`; the error kinds a failure crosses with |
| `profiling.py` | NVTX ranges and profiler windows Rust drives |
| `processing.py`, `rotary.py`, `tensors.py` | caller-owned image transforms and diffusion prompt framing; checkpoint rotary metadata normalized into typed recipes; numerical buffer requirements (`BufferConfig`) and borrowed tensor outputs |
| `runner` | the Python half of each runner, attention metadata, the execution context a runner prepares |
| `models` | capability classes, model inputs, component declarations, the checkpoint registry and the concrete models (`qwen3`, `qwen3_vl`, `siglip`, `bagel`, `sensenova_u1`, `minimax_h3`, `diffusion_gemma`, `stub`) |
| `loading` | checkpoint resolution, configuration normalization, weight mapping and a rank's model loading (`loading.rank`) |
| `nn`, `backends`, `bindings`, `cuda_graph` | layers, kernel providers, prepared operators and execution contexts, breakable CUDA graphs |
| `prefix_cache`, `diffusion`, `sampling`, `quantization`, `distributed`, `media` | prefix cache numerics, schedules and solvers, sampling, quantized tensors, partitioning and the process world, codecs |
