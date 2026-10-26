# Admin API, nodes and the manager

A deployment's leader daemon serves an admin API under `/admin` on its HTTP address, with the same API key as the OpenAI API. Everything a deployment can do while it serves goes through it: `uniserve serve` alone needs nothing else, and the optional `uniserve-manager` is one client of it among others. The design is `specs/live-updates.md`.

## Endpoints

| Endpoint | What it does |
| --- | --- |
| `GET /admin/status` | The deployment (name, model, engine state, paused, asleep, requests in flight, the engine's executable), the running and the last operation, and the next event's sequence number |
| `GET /admin/events?after=N&wait=S` | The events after sequence number `N`, waiting up to `S` seconds (at most 60) for one; `next` is the number to read after next, and `gap` says that events after `N` already left the log of 1024 |
| `POST /admin/update` | An update: `{"executable": PATH, "set": ["PATH=VALUE", ...], "wait": true}`. Assignments to the scheduling options alone (`max_running_requests`, `max_num_batched_tokens` and `scheduler_config`), within the batch the running workers were started for (at most as many calls and tokens a step and running requests), take effect in place at each scheduler's next step (`"in_place": true`); a change of policy reorders the waiting requests by it. Anything else is a hot update: a standby engine of that build and those options starts beside the running one, which drains; requests migrate to the standby, which serves. The options may change the scheduling, the CUDA graphs, FlashInfer and the workers' other options; an assignment that changes the model, the placement, the KV cache, the experts or how weights load is refused, since the standby maps the running engine's weights and caches. When the devices lack the memory the standby needs, the running engine first gives back the tail of its KV cache (below). `/engine/update` remains as its executable-only form |
| `POST /admin/pause_generation` | New requests wait; those in flight go on (SGLang's endpoint of that name) |
| `POST /admin/continue_generation` | New requests are served again |
| `POST /admin/flush_cache` | The engine revokes every cached prefix page; requests in flight keep theirs |
| `POST /admin/restart` | A hot restart of the daemon: `{"executable": PATH, "wait": true}`. A successor daemon of that build (this daemon's own when unset) takes the deployment over while this one finishes the requests it took (below); answers with the successor's process id once it serves |
| `POST /admin/shutdown` | Brings the deployment down as SIGTERM does, finishing what is in flight within `server_config.shutdown_timeout`, and releases its weights and caches at the host's Storage Server unless `{"retire": false}` |
| `POST /admin/update_weights` | `{"model_path": PATH}`: the engine rewrites its weights in place from another checkpoint of the served model, a local directory, a Hugging Face repository or a plugin's weight source `scheme://...` (`docs/plugins.md`) (SGLang's `update_weights_from_disk`; below) |
| `POST /admin/sleep`, `/admin/wake_up` | The engine gives back its KV cache and CUDA graph memory and keeps its weights, and restores both (vLLM's endpoints of those names; below) |
| `POST /admin/load_lora_adapter`, `/admin/unload_lora_adapter` | `{"lora_name": NAME, ...}`: the workers load a LoRA adapter into a slot of a plugin's adapter pool, or unload it (vLLM's endpoints of those names; below) |
| `POST /admin/cupti` | `{"attached": BOOL}`: every worker starts or stops CUPTI's watch of its waits for the device and record of its streams' device time, and answers whether CUPTI is attached (`attached`) and records the streams (`stream_activity`) (`docs/profiling.md`) |
| `POST /admin/hooks/NAME` | A worker hook a plugin adds (below) |

Each write is an operation: it has an identity, runs on a task of its own, and ends `succeeded` or `failed` in the status and in the event log, whether or not the client that asked stays connected. One operation runs at a time; another is answered `409`. `"wait": false` answers at once with `202` and the operation.

## Hot restarts

A daemon restarts in place without closing a connection or failing a request (`POST /admin/restart`, `crates/daemon/src/restart.rs`), as Envoy's hot restart does. It starts its successor (`uniserve serve --takeover`) with a socket pair between them. The successor loads the model's files and binds a request socket of its own; the daemon then stops its service thread between two rounds and hands the successor, as descriptors, its HTTP listener, its process socket with every engine and member connection it serves, its processes, the claim at the Storage Server and the connection its engine's process lives under, with the served spec, the admin's log and the video jobs; and it commands the engine to attach to the successor. From then on both accept on the listener: the engine serves both daemons, each its own requests. Once the successor serves, the daemon stops accepting, finishes the requests and video jobs it took (relaying each job's later changes and artifact to the successor), closes its engine connection without shutting the engine down, and exits. The engine and its workers are the same processes throughout. A process ends with its daemon through its lifeline, a pipe whose write end the daemon holds and whose read end the process inherits (`UNISERVE_LIFELINE`): the engine and the workers exit once they read the pipe's end, which arrives when the daemon ended however it did; a hot restart hands the write end to the successor with the processes, which it supervises from then on through their pidfds, so the pipe stays open across the restart.

A host's Storage Server restarts in place too (`uniserve storage restart [--executable PATH]`, `crates/storage-server/src/restart.rs`). The server starts its successor and, once no request, shoot-down or connector task is in flight, stops its connector and hands the successor its listening socket and its lock, every client's connection and pidfd, every tensor's memory and every page pool's memfd, with its state; the successor goes on with every connection and starts a connector of its own, and the server exits. Clients see no connection close; a request sent meanwhile waits in its connection.

## The KV cache's memory and the weights

A device KV cache is split along its units into eight segments, each of every layer's planes an allocation of its own mapped back to back at a fixed address (`crates/worker/src/memory/prefix_cache/planes.rs`), so that its tail can be given back to the device and created again in place while CUDA graphs keep their addresses.

- A hot update whose standby engine needs more device memory than is free shrinks the running engine's cache first. Each worker reports its segments, the device's free memory, the memory its process holds beside the weights and the cache, which the standby needs as much of, and the share of the device a parked standby keeps free beside it for its graph memory (`graph_storage_budget_bytes`, a tenth of the device); the scheduler withdraws the units of the segments that make the difference from allocation (the first segment always stays), ends the token requests holding them as migratable, so that the daemon submits them again with their generated tokens and they are recomputed on the units left, as vLLM's recompute preemption does, and once no request holds a withdrawn unit the workers give the tail back. The standby maps the segments left. Once the drained engine exited, the updated engine creates the tail again; a standby that does not start leaves the running engine to create it again. The shrink fails, before it withdraws anything, when the cache past its first segment is smaller than the difference.
- `/admin/sleep` holds new requests, recomputes the token requests in flight as a shrink does and lets the other requests finish; the workers then give back every segment and suspend their CUDA graph memory (`uniserve.sleep`). `/admin/wake_up` creates the cache again, restores the graph memory and lets the held requests through. A sleeping deployment refuses hot updates.
- `/admin/update_weights` checks that the daemon would process the new checkpoint's requests as the served model's (the same `tokenizer.json`, tokenizer settings, generation defaults and chat template, and the same model facts) and holds new requests; the engine recomputes the token requests in flight and lets the others finish, puts its workers to sleep, so that the cache's memory holds the new checkpoint as it loads, and has every worker load the checkpoint into new memory as it loaded the model and copy every tensor into place once all of them match in representation, shape and element type (`uniserve.loading.rank.update_weights`), then wakes them. A checkpoint that loads other tensors leaves the weights as they were. The workers rename their retained weights after the new checkpoint at the Storage Server, which a relaunch on it binds, and the engine flushes its prefix cache and keys host KV cache pages by the new weights. Weights streamed from host memory and DWDP experts are not updated in place: such a deployment is brought up again on the new checkpoint.

## LoRA adapters

The adapters are an extension point: a plugin keeps the adapter pool at fixed device addresses, allocated before the CUDA graphs are captured, and implements the worker hooks `load_lora_adapter` and `unload_lora_adapter`. `/admin/load_lora_adapter` assigns the adapter the lowest free slot of the pool and has every worker run `load_lora_adapter` with the request's body and the slot (`adapter`); from then on a request that names the adapter as its `model` computes with it, `/v1/models` lists it, and its prefix cache and encoder cache entries are kept apart from the base model's and every other adapter's. The worker writes each request's slot into its per-slot state at admission, and the causal LM runner fills, inside every forward, captured or not, the slot of every token's adapter into a fixed `[T]` tensor, which the plugin's kernels read through `uniserve.adapters.tokens()` (0: no adapter). An engine that replaces another, after a hot update or a relaunch, loads every adapter again before it serves; one that does not load leaves the registry, so requests naming it are refused. `/admin/unload_lora_adapter` removes the name at once and has every worker run `unload_lora_adapter` with the slot.

## Nodes

`uniserve node --host H --port P` runs a node's daemon serving no deployment. `POST /admin/deploy` with `{"config": TOML, "set": ["PATH=VALUE", ...]}` brings one up, as `uniserve serve --config` with those assignments would, and the node serves its API and admin routes on the same address once it starts; `GET /admin/node` reports `idle`, `starting`, `serving` or `stopping` and the last deployment's error. The deployment's `POST /admin/shutdown` brings it down and releases its memory, after which the node takes the next deployment. A node serves one single-node deployment at a time.

## The manager

`uniserve-manager` manages deployments across nodes through their admin API and depends on nothing else: a node never calls it or waits for it, so it attaches by reading a node's status and events and detaches by stopping.

```bash
uniserve-manager --node http://10.0.0.2:8000 --node http://10.0.0.3:8000 status
uniserve-manager --node http://10.0.0.2:8000 watch            # Ctrl-C detaches
uniserve-manager --node http://10.0.0.2:8000 deploy --config qwen3.toml
uniserve-manager --node http://10.0.0.2:8000 --node http://10.0.0.3:8000 update --set cuda_graph_config.decode_batch_sizes=1,2,4,8
uniserve-manager --node http://10.0.0.2:8000 shutdown
```

`update` is rolling: one node after another, stopping at the first that fails, so the other nodes serve while one updates. A change a node cannot make in place (a placement or KV cache change) is made by bringing a node down and up again with the new configuration while the other nodes serve.

## Worker hooks

A worker hook is a named operation that runs on every worker of the deployment at a step boundary, while no forward runs. A plugin implements it (`docs/plugins.md`):

```python
def register(registry):
    registry.hook("load_lora_adapter", load_lora_adapter)

def load_lora_adapter(payload, models):
    # payload: the request's JSON body; models: the modules the worker loaded
    ...
    return {"adapter": slot}
```

The endpoint answers with every worker's result. A hook no plugin implements fails with that reason. Hooks named `uniserve.*` are the engine's own (the cache's memory, sleep, wake, the weight update and CUPTI's attachment), which run through their endpoints alone; a plugin cannot register one.
