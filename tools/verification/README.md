# Verification

The formal models of UniServe's designs and the checks of the implementation against them. Each model lies in a folder of its own with its README: what it models, how to run its checks, and what they establish.

| Model | Design | Implementation checked against it |
| --- | --- | --- |
| [`buffer-pool/`](buffer-pool/README.md) | The Storage Server's buffer pool (`specs/buffer-pool.md`): its handlers, deferred work, copies, tiers, regions, epochs on trial, an engine's loss, crossings, the handoff's lends, imports and borrows | `crates/storage-server/src/buffer_pool` (the `verification` feature's `buffer-pool-verify`): an enumeration of small configurations, replays of serving runs' traces, and mutations of the rules |
| [`data-plane/`](data-plane/README.md) | The KV data plane across replicas and nodes (`specs/kv-data-plane.md`): content keys on the device, host and Mooncake tiers, the cluster directory and its stale account, the prefill-to-decode handoff by push, leased read and pull by key, tiering moves, with role and epoch switches of the configuration's convergence as its environment | Not yet: the directory, the leased read and the pull by key are not implemented; `specs/kv-data-plane.md` §6 maps each variable and action to its code or target place |

Two designs are modelled. The data plane's model takes each node's buffer pool as an environment that meets the buffer pool model's P1 and P2, and the configuration's convergence as an environment that meets its assumptions E1 to E4. A model is a folder with `model/` (the design and its neighbours), `properties/` (what must hold), `proof/` (the proofs, where there are any), `configs/` (what the model checker searches), `scripts/` (the runners) and `log.md` (each run's prediction, written before it, and its outcome).
