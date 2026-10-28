# Execution profiling

A serving deployment profiles itself without restarting a process or pausing service. Every worker subscribes its process to NVIDIA's CUPTI once, as it starts, and every collector below shares that subscription: two counters and the fence probes are always on and exported as Prometheus metrics, and a profiling session (`/admin/profile`) attaches further collectors to the executing world at a step boundary and detaches them at another, writing one SQLite file per worker that `uniserve_eval.profile` reads and converts to a Perfetto trace. Captures of Nsight Systems and cProfile are read by the execution sanitizer, which produces hints about host waits, GIL ownership, copies and language crossings. Neither changes execution, rejects a request or establishes that an operation is unnecessary: inspect each finding in its workload context, then measure any optimization without the diagnostic instrumentation.

## CUPTI in a worker

A worker loads the CUPTI library that sits beside the CUDA runtime it loaded, which is the `nvidia-cuda-cupti` package of its Python environment; the CUDA 13.0 toolkit's own CUPTI (2025.3.0) stalls a worker that replays CUDA graphs while kernels are recorded. CUPTI accepts one subscriber per process, so the worker subscribes once, before it loads its model, and the counters and every session collector enable the callbacks and activity records they need on that subscription and release them when they stop. A worker whose process a profiler already subscribed, such as Nsight Systems tracing CUDA, logs that CUPTI is not subscribed: it reports neither counter, and a session reports every CUPTI collector unavailable on it; the fence probes do not use CUPTI and stay on. The CUPTI library stays loaded in the process: removing it (`cuptiFinalize`) after it recorded a serving worker's activity leaves the device work the worker submits afterwards waiting forever.

CUDA graph executables are instantiated in their current CUPTI tracing mode: whole-graph records (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`) or node records. When a timeline changes this mode, the worker replaces each affected executable before its next replay, using the retained source graph. This preserves captured tensors, allocator pools and random-number state without recapturing computation or synchronizing the device. Executables instantiated for whole-graph tracing must not run in node mode: with `nvidia-cuda-cupti` 13.0.85 on GB200, that combination retains timestamp slots until CUPTI exhausts its profiling buffers.

## Watch host waits and stream idle time while serving

A serving worker reports two counters from CUPTI with every batch result, which the daemon exports as Prometheus metrics:

| Metric | Labels | What it counts |
| --- | --- | --- |
| `uniserve:worker_blocking_calls` | `thread`, `call` | Driver calls that made a serving thread wait for the device: stream, event and context synchronization, synchronous copies, `cuMemFree`, `cuMemFreeHost` and `cuMemUnmap`. Runtime and PyTorch calls reach these driver calls and are counted too. `thread` is `builder` (the jobs it performs, the executor's Python calls among them) or `control` |
| `uniserve:worker_stream_us` | `stream`, `state` | Microseconds of the device time of each observed stream, `device` (the primary stream) and `copy` (the host KV cache's copy stream), from CUPTI's activity records of the kernels, copies and fills on the stream, graph nodes included (a graph's launch is one record while a session traces graphs whole): `busy` is the length of the union of their intervals, `gap` the time the stream had nothing to run between its first operation and its latest end, whether it waited for the host, for another stream or for any work, and `late` the time of records that arrived after the two seconds of device time the union keeps behind the stream's latest end, counted in neither |

A serving call of the executor (`start`, `prepare`, `execute`, `commit`, `join_expert_step`, `release`) that makes the host wait for the device fails with `InvariantViolation`, naming the calls. The rest of a job and the control thread are counted without failing, since some of their reads still wait for their producers. Both counters stay on while the worker serves.

## Fence probes

Fence probes time the model's components on the device, inside CUDA graphs, at every step. Every module class of the numerical library declares its component as the `<component>` of its function (`function = "<component>/<revision>"`, `uniserve/numerics.py`); a worker attaches a probe at the entry and one at the return of every module whose component matches `profiling_config.fence_components` (`docs/configuration.md`). A probe is a single-thread kernel launched on the module's current stream: a forward captured into a CUDA graph at warmup captures its probes, which each replay runs. The worker brackets each batch's execution with step markers of the same kernel on the computation's stream, which store the batch's id where the probes between them read it, so every record names its step.

A probe first reads a device switch, a word per component and a master word in device memory; off, it returns at once. On, it reads `%globaltimer` and writes its time, its probe and its step into the next slot of a double buffer in pinned host memory the worker allocated, 4096 records of 16 bytes a half; the probe that completes a half publishes it. The worker's control thread reads every complete half, and at most once a second the half being filled, without any device synchronization; a record tags its half's fill, so one the device overwrote before the host read it is counted as lost (`uniserve:worker_fence_records_lost`). At each step's end marker the worker pairs each module instance's entry and exit in time order, sums the durations of a component's instances in the step, and counts the sum in the histogram of the step's stage and the component, beside the step's span between its markers (component `step`):

| Metric | Labels | What it counts |
| --- | --- | --- |
| `uniserve:worker_fence_step_seconds` | `stage`, `component` | A histogram of per-step durations on the device: for each step of the stage (its call kind, such as `prefill` or `decode`), the summed time of the component's probed instances, or the step's span for `step`; buckets from 10 µs to 10 s |
| `uniserve:worker_fence_records_lost` | | Probe records overwritten before the worker read them |

**Cost.** Every probe is a kernel launch in each replay of a captured graph, whether it records or not. The always-on budget is LMTracer's reported 0.58%. On GB200 with Qwen3-30B-A3B in bfloat16 (decode of 512 tokens per request, 60 s per point, probes switched off and on alternately in one server, four pairs per point), the step time measured:

| Probed components | Probes per step | On against off, 1 request | On against off, 32 requests | Off against no probe |
| --- | --- | --- | --- | --- |
| `*.Transformer` (the default) | 2 | −0.73%, −0.19% | −3.5%, +5.8% | within the ±5–10% spread between servers |
| `*.TransformerLayer` | 96 | +1.71% (+0.79 to +2.76%) | +1.15% (−3.54 to +3.32%) | within the spread between servers |
| `*` | 2028 | +18.4% | +18.6% | +21% to +36% |

Each probe thus costs about a microsecond of every step even when switched off, and about another microsecond when it records, so the default probes each model's decoder stack only; probing every layer costs 1–2% and every component 15–35%.

`profiling_config.fence_probes = false` switches every probe off in place: `PUT /admin/config` with that assignment has every worker write its device switch at the engine's next step boundary, and the graphs keep their probes, each of which then returns at once. Changing `fence_components` changes which modules carry probes in the captured graphs, which is a new world (`docs/admin.md`).

## Profiling sessions

A session attaches collectors to every worker of the executing world for a window of steps or seconds, and leaves one SQLite file per worker in a directory it names. One session runs at a time:

```bash
curl -X POST http://127.0.0.1:8000/admin/profile -H 'Content-Type: application/json' -d '{
  "collectors": ["timeline", "fence_timeline", "pm_sampling"],
  "stages": ["decode"], "start_step": 10, "steps": 200,
  "output_dir": "/data/profiles", "max_slowdown": 0.1}'
curl http://127.0.0.1:8000/admin/profile             # its state
curl -X DELETE http://127.0.0.1:8000/admin/profile   # end it now
```

| Field | Default | Meaning |
| --- | --- | --- |
| `collectors` | Required | The collectors below, at least one |
| `steps` | Unbounded | The steps of the selected stages each worker records |
| `start_step` | `0` | The steps of the selected stages each worker skips before its window opens |
| `seconds` | Unbounded | How long the session lasts from its start |
| `stages` | Every stage | The steps recorded, by call kind (`prefill`, `decode`, `verify`, `denoising`, ...) |
| `workers` | Every worker | The workers recorded, by rank |
| `components` | Every probed component | Name patterns (`*` matches any run) of the components whose fence records `fence_timeline` keeps |
| `output_dir` | Required | The directory each worker writes `profile-SESSION-worldWORLD-rankRANK.sqlite` into; it is created |
| `timeline`, `pm_sampling`, `range_metrics`, `pc_sampling` | Below | Each collector's options |
| `max_slowdown` | Unguarded | The quality-of-service guard's bound, a fraction of the step time before the session |

The session starts at the engine's next step boundary: the workers prepare their collectors and create their files, and answer with each collector's state, `prepared` or `unavailable` with the reason (a worker without CUPTI's subscription, a metric set that needs more than one pass). Each worker arms its collectors before the forward of the first step its window takes and disarms them after the forward of the last, as jobs that step's forward waits for, so that what CUPTI records begins and ends at step boundaries; it disarms them before a step of another stage and arms them again before the next selected one. A step is one batch's forward; every step during the session is listed in the file with whether the window took it. A session ends when every selected worker's window of steps ended, its seconds passed, the operator ends it, the quality-of-service guard ends it, or its world stops executing: a world change ends it before it parks the serving world, and a rollback ends the session of the world on trial; neither carries a session into the next world. Ending has every worker of a serving world disarm, release its collectors, write what remains and close its file.

`GET /admin/profile` answers with the running session, or the last that ended: its state, each worker's collectors and, once it ended, why and each worker's file and its rows by table; the steps and the bytes the workers recorded so far (`steps`, `bytes_written`), reported with their batch results; the step time before the session and over the session's last 10 seconds (`overhead`); and the quality-of-service guard's actions.

| Collector | Mechanism | Options | What its file holds |
| --- | --- | --- | --- |
| `timeline` | CUPTI's activity records of kernels (`CONCURRENT_KERNEL`, which keeps kernels concurrent), copies, fills, NVTX marks and their data, memory allocations and the NVLink and PCIe topology, and the callbacks of every runtime and driver API call; a CUDA graph's launch is one record, or one per node | `graphs`: `whole` (default) or `nodes` | `kernel`, `copy`, `fill`, `graph`, `marker`, `marker_data`, `memory`, `nvlink`, `pcie`, `api` |
| `fence_timeline` | Every fence probe record the worker reads, of the selected components and the step markers | | `fence`, with `fence_probe` |
| `pm_sampling` | CUPTI's PM sampling: hardware counters of the whole device every interval of device time, in one pass | `metrics` (PM sampling metric names; default `gr__cycles_active.avg`, `sm__cycles_active.avg`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`), `interval_ns` (default 100000, at least 20000) | `pm_sample` |
| `range_metrics` | CUPTI's range profiler with metrics one pass collects, never replaying work | `metrics` (default `gpu__time_duration.sum`), `ranges`: `kernel` (a range per launch, a CUDA graph's launch being one) or `component` (a range per probed module instance's call, which only calls outside a CUDA graph's replay push: a replay runs no host code), `pass_ranges` (the ranges one pass holds, at least 32; default 512) | `range_metric` |
| `pc_sampling` | CUPTI's PC sampling in continuous mode, which samples warps without serializing kernels | `period`: each SM samples a warp every `2^period` cycles, from 12 to 31 (default 17) | `pc_sample`, `pc_take`, `pc_stall_reason` |

**Sampler reads.** A worker reads its armed samplers every 100 ms on a thread of their own: PM samples, which it decodes and evaluates, and PC samples. Steps do not wait for a read, except that arming or disarming the collectors, which a step's forward waits for, waits for a read in progress. A PM read decodes into one counter data image, which CUPTI fills no further: the records past a full image are lost (on GB200 with `nvidia-cuda-cupti` 13.0 no later read finds them). The image and PM sampling's hardware buffer, both process host storage, therefore hold twice the samples of 100 ms at the session's interval, the buffer at the image's bytes per sample: on GB200 with the default metrics, 13.6 KB of image per sample, against the 8 KiB of hardware buffer a sample was measured to take. A read that fills its image, and a buffer that wraps before a read (which can also fail the next read, `CUPTI_ERROR_UNKNOWN`), lose samples, which the worker logs. Decoding and evaluating a sample of the default metrics takes about 14 µs on GB200's CPU: at an interval of 20 µs, reads every 100 ms kept every sample over 5 s and one thread about 70% busy; at 10 µs the reads fell behind and lost samples. Intervals below 20 µs are therefore refused. PC sampling keeps its records in CUPTI's hardware buffer, sized per period (below), which a read empties; when it fills, the samples are lost and the read fails, which the worker logs.

**Range profiler passes.** `range_metrics` collects into passes of `pass_ranges` ranges and reads each every `pass_ranges / 32` steps the window records, so a pass holds up to 32 kernel and graph launches per step. A pass takes 1,421,312 bytes of device storage per range, rounded up to 2 MiB, the same for one to four metrics and either range mode (measured on GB200 with `nvidia-cuda-cupti` 13.0); CUPTI takes it at the pass's first start and keeps it until the collector is released. The storage is booked in the GPU's ledger (below), so the default of 512 ranges is chosen by its storage cost:

| `pass_ranges` | 512 (default) | 2048 | 65536 |
| --- | --- | --- | --- |
| Device storage | 727,711,744 bytes (0.73 GB) | 2,910,846,976 bytes (2.9 GB) | 93,147,103,232 bytes (93 GB) |
| Read every | 16 steps | 64 steps | 2048 steps |

CUPTI also needs free device addresses among the lowest ones for a pass's storage: on GB200, address reservations without a hint that take the addresses up to 1 TiB leave a pass of 2048 ranges unable to start (`CUPTI_ERROR_UNKNOWN`), whatever the device has free. A worker's own reservations (its graph pools and the mappings of its weights and KV cache) are therefore placed from 4 TiB on. PyTorch's expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the ranks' default) still reserve from the lowest addresses, 0.9 to 1.1 TB of a warmed-up Qwen3-30B-A3B rank's address space; beside them, a pass of 65,536 ranges starts on such a rank after warmup, with 128 GB of the device free, where with the rank's own reservations among the lowest addresses too no pass past 1,407 ranges started.

**Device storage of a session.** A `range_metrics` session books its pass's device storage in the storage ledger of each selected worker's GPU, with the worker's process item, while it runs. Where a ledger lacks the room (a KV cache sized by the quota leaves none), the KV cache first gives back tail segments, as a world change's preparation does: the requests holding them run on, and the session starts once they let them go. The cache grows back once the session ends. A session for which even every segment past the first leaves too little room is refused, naming the bytes and every ledger entry.

**Collectors that share a device.** `pm_sampling`, `range_metrics` and `pc_sampling` each hold the device's performance monitors while they run, and CUPTI gives the monitors to one profiler at a time: on GB200 with `nvidia-cuda-cupti` 13.0, whichever of two starts second fails (`CUPTI_ERROR_HARDWARE_BUSY`, or an invalid parameter between the range profiler and PC sampling). A session therefore runs at most one of them, and a request naming two is refused; `timeline` and `fence_timeline` run beside any of them. CUPTI also refuses to enable activity records while PC sampling is enabled (`CUPTI_ERROR_NOT_COMPATIBLE`), so a worker enables PC sampling each time it arms its collectors, after the timeline started, and disables it each time it disarms them, before the timeline stops.

**PC sampling and CUDA graphs.** CUPTI attributes the samples of a CUDA graph's kernels to their program counters only when PC sampling was enabled when the graph was captured, and stayed enabled since; otherwise it counts them as samples of non-user kernels, without records (on GB200 with `nvidia-cuda-cupti` 13.0, over replays of a captured graph: every one of 1.8 million samples, against all but about 1,300 when PC sampling was enabled before the capture; instantiating the graph again under PC sampling does not change it). A worker captures its graphs at warmup, so by default a session's own sampler attributes the kernels launched outside CUDA graphs, and `pc_take` counts the samples of graph kernels among its non-user samples. A deployment that profiles its graphs' kernels sets `profiling_config.graph_pc_sampling` (`docs/configuration.md`): every worker then enables PC sampling, stopped, before it captures its graphs and keeps it enabled for its life (having first enabled the activity kinds of its always-on stream record, which CUPTI refuses to enable once PC sampling is), and a session only sets the sampler's period and buffer at its first arm, starts it at each arm and stops it at each disarm. While PC sampling is enabled CUPTI refuses PM sampling (`CUPTI_ERROR_HARDWARE_BUSY`), the range profiler and the whole-graph activity record (`CUPTI_ERROR_NOT_COMPATIBLE`), so such a deployment refuses sessions of `pm_sampling`, `range_metrics` and timelines that trace graphs whole, naming the option; timelines with `"graphs": "nodes"` run.

**PC sampling buffer.** CUPTI keeps the samples in a hardware buffer of process host storage, which its own thread empties; a buffer too small for the sampling rate fills, and its samples are lost. A worker sizes the buffer for the session's period from a measurement on GB200 with every SM busy: a size holds when no window of 0.6 s fills it, each period takes twice the smallest size that held three windows (capped at CUPTI's 512 MiB), and the size taken held ten more. Periods below 12, which no buffer of up to 512 MiB holds, are refused:

| `period` | 19 to 31 | 18 | 17 | 16 | 15 | 14 | 13 | 12 | 11 and below |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smallest size that held | 1 MiB | 2 MiB | 3 MiB | 16 MiB | 16 MiB | 64 MiB | 128 MiB | 512 MiB | none (512 MiB fills) |
| Buffer | 2 MiB | 4 MiB | 8 MiB | 32 MiB | 32 MiB | 128 MiB | 256 MiB | 512 MiB | refused |

**Quality-of-service guard.** The daemon averages the executing world's step time over the minute before the session and, every half second, over the session's last 10 seconds. With `max_slowdown`, a step time past the earlier one by more than that fraction turns off the costliest collector still running, at most once every 10 seconds so that each change is measured; a slowdown that remains with no collector left ends the session. The collectors are turned off in this order: `range_metrics`, `pc_sampling`, `timeline`, `pm_sampling`, `fence_timeline`. Every action, with the slowdown that caused it, is in the session's state.

**NVTX.** The worker's `uniserve.worker.*` ranges and the library's ranges follow a run-time switch, which starts as `UNISERVE_NVTX` states and which a `timeline` collector turns on while it records and restores after. NVTX reaches CUPTI only through the injection library the process names in `NVTX_INJECTION64_PATH` at its first NVTX call (without it CUPTI records no mark), so a worker names its environment's CUPTI there as it starts, unless its environment names another library, as Nsight Systems does.

### Session files

Each worker writes one SQLite file per session, in UniServe's own schema: every raw record kind has a table, beside `metadata` (the session, the world, the rank, the device, its chip, the CUPTI library and version, the request, and `doc.*` entries stating what every table's identifiers and clocks mean), `strings` (kernel, API, NVTX and domain names by id), `clocks`, `steps` and `collectors` (each collector's state over the session, the guard's releases among them). The schema states its version (`schema_version`) and changes only by additions. A writer thread of the worker receives the records over a channel, commits them every 200 ms, and checkpoints the write-ahead log into the file when the session ends; a worker that exits during a session leaves what it committed in the file and its log.

Every time is in nanoseconds of one of two clocks: CUPTI's (about `CLOCK_REALTIME`), for CUPTI's records, the steps and PM samples; and each device's `%globaltimer`, for the fence records. `clocks` holds samples of them at the session's start and end: CUPTI's clock with `CLOCK_REALTIME` and `CLOCK_MONOTONIC`, and each device's `%globaltimer`, read by a kernel that stores it into pinned host memory while the host times its appearance, with half that time as the sample's uncertainty. A client aligns a clock through the sample of least uncertainty. CUPTI's records carry the correlation id of the API call that issued them, a graph's nodes that of its launch.

```bash
.venv/bin/python -m uniserve_eval.profile summary /data/profiles/profile-1-world1-rank0.sqlite
.venv/bin/python -m uniserve_eval.profile perfetto /data/profiles/profile-1-*.sqlite -o trace.json
```

`summary` states each file's session, worker, collectors and rows by table. `perfetto` writes one trace of a session's files in the Trace Event Format, which the Perfetto UI opens: a process per worker, with a track per CUDA stream, host thread and device's fence probes (each probed component instance's call a slice, nested as the calls nest), the steps, and counter tracks for PM samples, all on CUPTI's clock. `uniserve_eval.profile.ProfileFile` reads a file from Python.

## Capture and inspect a CUDA timeline with Nsight Systems

Set `UNISERVE_NVTX=1` before starting the worker to start its ranges on. The native executor emits synchronous ranges named `uniserve.worker.admit`, `prepare`, `inputs`, `execute`, `retire`, `poll`, `result` and `close`, with the batch ID as a numeric payload. Shared resource reclamation uses `uniserve.worker.reap` without a batch payload. Startup phases emit `uniserve.worker.describe`, `load`, `plan`, `bind`, `warmup` and `capture` without a payload. Python numerical ranges, such as `uniserve.collective` launches, remain nested within those stages. Native ranges use the NVIDIA NVTX SDK directly and issue no Python callbacks or CUDA operations. Nsight Systems holds CUPTI's subscription in the processes it traces, so their sessions run no CUPTI collector.

Capture a representative workload with Nsight Systems. `CAPTURE` is an output prefix and `WORKLOAD` is the caller's Python entry point; select both explicitly. For serving, replace the Python command with the normal `uniserve serve` command and run the fixed workload against it.

```bash
UNISERVE_NVTX=1 nsys profile \
  --trace=cuda,nvtx,python-gil --cuda-trace-all-apis=true \
  --cuda-graph-trace=graph --sample=process-tree \
  --cudabacktrace=sync:0,memory:0 --python-backtrace=cuda \
  --export=sqlite --output="$CAPTURE" \
  .venv/bin/python "$WORKLOAD"

.venv/bin/python -m uniserve_eval.sanitizer trace "$CAPTURE.sqlite"
.venv/bin/python -m uniserve_eval.sanitizer trace "$CAPTURE.sqlite" --json
```

The reader uses the Nsight Systems 2025.4 SQLite export schema. It opens the file read-only. `--scope PREFIX` selects synchronous NVTX ranges by name, defaulting to `uniserve.`; repeated `--rule NAME` options select individual rules. Hints do not cause a failing exit status. Invalid inputs or unreadable exports do.

| Rule | Execution expectation | Evidence and interpretation |
| --- | --- | --- |
| `host-sync` | Batch advancement should enqueue work or query readiness without waiting for unrelated device completion. | A Runtime or Driver device, context, stream or event synchronization call occurred inside the selected scope. Immediate result consumption, initialization, shutdown and failure cleanup may require it. Stream event waits and queries are not host waits. |
| `gil-held-wait` | A native GPU wait should release the GIL so independent Python work can proceed. | A CUDA host synchronization call overlaps a GIL ownership interval on the same thread. The hint reports overlap duration and retains any other threads in the process waiting for the GIL during that overlap. No duration threshold establishes harm; trace overhead, interpreter identity and workload dependencies limit the conclusion. |
| `sync-copy` | Host/device copies should overlap independent work when their consumer allows it. | A synchronous copy API has a correlated host/device transfer. An immediately consumed result or startup transfer can justify synchronous execution. |
| `pageable-copy` | Repeated asynchronous host/device transfers should use reusable pinned host buffers when useful. | A correlated copy uses pageable host memory. An Async API may stage or block; a one-time transfer may not justify pinning. |
| `device-roundtrip` | Intermediate numerical values and device decisions should stay on device when practical. | A D→H copy completes before an H→D submission on the same thread, device and NVTX scope. This ordering does **not** establish that the copies carry the same data; independent inputs or required CPU computation can explain it. |

Each hint retains API names, nanosecond intervals, global thread and CUDA correlation IDs, enclosing scope, batch number when available, native stacks when captured, and correlated copy sizes, streams and memory kinds. Nested numerical scopes inherit the native batch number. Backtrace activity records contribute stacks to their API call rather than another invocation. Copy correlation includes the process, keeping rank-local correlation IDs separate.

GIL evidence comes from Nsight's `GIL Trace` NVTX domain, including the original holding and waiting intervals. The reader matches domains within each process and ownership on the CUDA caller's thread. It omits unclosed intervals and reports the coverage gap. Missing GIL tracing means ownership is unknown; CUDA wait duration and cProfile cannot supply it. A captured waiter establishes temporal contention evidence, not a proven deadlock or a dependency on that GPU operation. The export does not identify individual Python interpreter locks. CPU-only GIL contention, mutex waits and callback dependencies are outside this rule's coverage.

CUDA Graph execution does not itself need the GIL once submitted. Python preparation, submission, callbacks, result handling and final-reference destruction can still acquire or hold it. Use the same GIL rule for graph and eager workloads; it describes observed waits rather than assuming that graph replay eliminates language interaction. Nsight's [Python GIL tracing](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-gil-tracing) supports CPython 3.9 or later on Arm SBSA, Linux x86 and Windows, excluding interpreters built with `Py_GIL_DISABLED=1`.

Coverage counts describe recorded activities. A missing activity, missing stack or unmatched scope is a coverage gap. No hints does not mean no unnecessary synchronization. GPU graph-level capture does not expose every graph node; Python metadata accesses and dependency relationships between independent batches are not inferred from CUDA timing. Use Nsight's timeline and native reports for GPU gaps, utilization and stream dependencies.

For bounded worker capture, the existing `UNISERVE_TORCH_PROFILER_DIR`, `UNISERVE_PROFILE_START_STEP`, `UNISERVE_PROFILE_STEPS` and `UNISERVE_CUDA_PROFILER` settings can drive `--capture-range=cudaProfilerApi`. This window brackets numerical execution steps: the first batch's earlier preparation and the last batch's later completion may lie outside it. Capture the whole workload when those phases are under investigation. Stacks and detailed tracing add overhead; keep capture duration proportionate to the question.

CUDA's [synchronization semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) depend on operation and memory kind. Nsight's [analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html) describes the export and built-in reports. The sanitizer's rules are advisory interpretations of those activities.

## Inspect Python/native call edges

Standard cProfile captures some Python/native calls and callbacks. Select the native function names to inspect; the analyzer matches builtin labels, with no assumptions about installation paths.

```bash
.venv/bin/python -m cProfile -o "$CALLS" "$WORKLOAD"
.venv/bin/python -m uniserve_eval.sanitizer calls "$CALLS" \
  --native "$NATIVE"
```

`NATIVE` is a substring of the builtin's cProfile label, such as `torch._C`; repeat `--native` to select more. cProfile does not record TVM-FFI function calls, so the calls between the worker runtime and its Python numerical functions never appear in this report. Read those crossings from the `uniserve.worker.*` ranges of the CUDA timeline.

The report shows repeated edges in both directions, call counts and inclusive time. Use it to locate repeated metadata conversion or per-row callbacks for further investigation. A repeated batch numerical call is expected. Inclusive time includes the callee's work and must not be summed or interpreted as FFI overhead.

cProfile supplies no temporal ordering or batch attribution. Its capture does not automatically cover native worker threads; many C API attribute accesses and extension constructors are absent. The report lists the selected native functions actually captured and reports an unknown coverage result when none match. Use this evidence alongside the CUDA timeline, not as a complete crossing count.
