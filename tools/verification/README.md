# Verification

The formal models of UniServe's designs and the checks of the implementation against them. Each model lies in a folder of its own with its README: what it models, how to run its checks, and what they establish.

| Model | Design | Implementation checked against it |
| --- | --- | --- |
| [`buffer-pool/`](buffer-pool/README.md) | The Storage Server's buffer pool (`specs/buffer-pool.md`): its handlers, deferred work, copies, tiers, regions, epochs on trial, an engine's loss, crossings | `crates/storage-server/src/buffer_pool` (the `verification` feature's `buffer-pool-verify`): an enumeration of small configurations, replays of serving runs' traces, and mutations of the rules |

The buffer pool is the one design modelled so far. A model is a folder with `model/` (the design and its neighbours), `properties/` (what must hold, pinned), `proof/` (the proofs), `configs/` (what the model checker searches), `scripts/` (the runners) and `log.md` (each run's prediction, written before it, and its outcome).
