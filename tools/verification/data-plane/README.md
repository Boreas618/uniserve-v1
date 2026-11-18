# The KV data plane's design model

The model of `specs/kv-data-plane.md`: the KV cache stack's data plane across replicas and nodes. Pages are addressed by content key on three tiers (a replica's device memory, its node's host tier, Mooncake Store); a cluster directory summarises what each node holds and may lag it; a prefill-to-decode handoff moves a request's pages by push into pages the decode reserved, by a read of the prefill's device pages under a lease, or by a pull by key from the host tier or the store, and its unkeyed items (the tail after the last full page, the live recurrent state) by push or read alone; the KV program's tiering moves copy pages between tiers on the directory's account; and the configuration's convergence switches roles and epochs at any step. Nothing here needs a device.

## Layout

| Folder | What it holds |
| --- | --- |
| `model/` | `KvPlane.tla`, the design model and its environment: the convergence's role and epoch switches under the assumptions E1 to E4, Mooncake Store under M1, each node's buffer pool under B1 (`specs/kv-data-plane.md` §5). Each hazard is behaviour switched on by its name in `Hazards`: `StaleAddress`, `EarlyRelease`, `Unfenced`, `UnreservedPush`, `KeyedTail`, `SeedBlind`, `SeedlessKey`, `RoleFlush` (`specs/kv-data-plane.md` §3) |
| `properties/` | `PlaneSafety.tla`: D1 to D7 and G1 as formulas over `KvPlane` (D1 is the stack's core invariant: every content a request reads is what its key, or its live state, names); `PlaneFinality.tla`: D8 under weak fairness of each request's own steps; `PlaneWitness.tla`: behaviours the design must reach, each stated as never happening so that TLC reports it; `properties.toml` lists them |
| `configs/` | The configurations TLC searches (`MC_*.tla`, `MC_*.cfg`) |
| `scripts/` | `tlc.sh`, one configuration; `suite.sh`, every run with the verdict expected of it |
| `log.md` | Every run with the prediction written before it and its outcome |

The tools are TLC (`tla2tools.jar` 1.8.0) under a JDK 21, in `TLA_TOOLS` (default `/workspace/dev/tools/tla`). A configuration runs from `configs/` with `model/` and `properties/` as TLC's library. TLC's states and logs go to `$TLC_LOGS`, by default `artifacts/verification/data-plane/` under the repository, and the states are removed when a run ends.

| Configuration | What it searches |
| --- | --- |
| `MC_Handoff` | One request (a full page, a tail) from a prefill replica on one node to a decode replica on another: every path, the directory reported and lagging, three tiering moves, every eviction |
| `MC_Reuse` | Two requests of one prefix (a full page, a tail) through one prefill and one decode replica on one node: reuse of a cached page on either side, a request's pages taking pages another released or aborted, a pull from the store after one move puts the page there |
| `MC_Stale` | Two requests of two prefixes, full pages alone, one host page and two moves: the directory reports a page under one key while it comes to hold the other; the store's events lag |
| `MC_Switch` | Two requests of one prefix on two replicas on two nodes while up to two role or epoch switches happen at any step, over two seeds; the device tier alone |
| `MC_Finality` | `MC_Handoff` without symmetry, under `FairSpec`, for D8 |

## Running the checks

```bash
tools/verification/data-plane/scripts/suite.sh                     # every part
tools/verification/data-plane/scripts/suite.sh hazards witnesses   # the quick parts
tools/verification/data-plane/scripts/tlc.sh MC_Handoff
HAZARDS=StaleAddress tools/verification/data-plane/scripts/tlc.sh MC_Stale
ONLY="NoPullStore" tools/verification/data-plane/scripts/tlc.sh MC_Handoff
```

`TLC_WORKERS` and `TLC_HEAP` set TLC's workers and its Java heap (defaults 32 and 32g); `suite.sh` runs the design's configurations side by side, so their sum is what the host must hold. `MC_Switch` takes 14 minutes on 120 workers and a 200 GB heap (`TLC_WORKERS=120 TLC_HEAP=200g tlc.sh MC_Switch`); the hazards and witnesses each stop at their violation within seconds.

## What is established

Terms. Checked: no violation in a complete search of a stated configuration. A run that times out or stops at a bound is INCONCLUSIVE. Nothing in this model is proved. `log.md` holds each claim's runs and figures.

- **Checked (TLC).** D1 to D7 and G1 hold in `MC_Handoff` (432,646 distinct states), `MC_Stale` (12,979,546), `MC_Reuse` (22,244,987) and `MC_Switch` (116,331,783); D8 holds in `MC_Finality` (1,726,447), with D1 to D7 (`log.md` entries 2, 5, 7 and 9).
- **Each guard is needed.** Each hazard violates the property `suite.sh` states for it: `EarlyRelease` D4, `UnreservedPush` D3, `KeyedTail` D6, `Unfenced` D1, `StaleAddress` D2 or D1, `SeedBlind` D1, `SeedlessKey` D1, `RoleFlush` D3, D4 or D5; and, checked alone, the wrong content itself (D1) for `KeyedTail`, `EarlyRelease`, `UnreservedPush` and `RoleFlush`.
- **Not vacuous.** Every witness is violated: each path delivers a page, a read delivers the tail, a pull the directory chose misses and is computed, a read whose lease expired is computed, a request reuses another's cached page, a handoff between replicas of two seeds decodes, a request decodes on a replica switched away from decoding.
- **Not established.** That the implementation refines the model: the directory, the direct read with its lease and the pull by key are not implemented yet, and no trace of a serving run is checked against this model. `specs/kv-data-plane.md` §6 maps each variable and action to its code or its target place. Head-range resharding between layouts, device prefetch and the transports' internals are outside the model (`specs/kv-data-plane.md` §1).
