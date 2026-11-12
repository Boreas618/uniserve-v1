# Predictions

Each entry is written before the run it names and is not edited afterwards; the outcome is added beneath it. A run that times out or stops at a bound is INCONCLUSIVE.

## 1. The core of the design, one group (`MC_Core`), 2026-11-12

- **Model:** `KvCore` with `Attach`, `Allocate`, `Pin`, `Evict`, `Release`, `Detach`, `Commit`, the executor of A1 to A4 and a control plane that issues any request and hands on or drops any completion at any time. All rules in force. Two units, two slot rows, one key, two owners, one group, one position, two batches, two requests under way.
- **Prediction:** TLC reports a violation of `WriteIsolation` (P1). The expected behaviour: an owner is attached and allocated a block; its `Detach` is taken while no executor holds its entries, so it carries no stamp and its units are reclaimed at once; the completion of its `Allocate` is handed to the executor afterwards; the unit is allocated to a second owner; a batch writes it for the first. A1 to A4 as written allow it: the runs came from a completion and were written at the position the request named.
- **If it holds:** the assumptions do not carry "the scheduler's decisions cannot break them" (`specs/buffer-pool.md` §9), and A1 has to state when a completion may still be handed on.
- **Outcome:** as predicted. `P1` is violated in seven states: `Attach(o1)`, `Allocate(o1, p1)`, a `Release` of that position issued and taken with no stamp because no executor held the entry, the `Allocated` completion handed to the executor afterwards, and a batch launched that writes the reclaimed unit. Search stopped at the violation (34,199 states generated).
- **Consequence:** A1 states when a completion may be handed on: only before any `Release`, `Detach` or `Commit` that names its entry is handed to an executor; a completion of a request submitted before such a request is dropped. `KvCore` carries it as the guard of `ExecMap` and as `Kept` in the handlers of stamped requests.

## 2. The core with A1 as restated (`MC_Core`), 2026-11-12

- **Model:** as entry 1, with the restated A1.
- **Prediction:** the search completes with no violation of `TypeOK`, `P1`, `P2`, `P3`.
- **Outcome:** not as predicted. `P1` is violated in ten states: a `Release` of an owner's position and a `Detach` of the owner are both under way; the executor applies the `Release` and stamps it; the `Detach` then finds no entry in the executor's table, carries no stamp, is taken first and reclaims the unit while the batch that writes it is in flight. The model let a group stamp a request only while its table held the entries concerned, which is how `specs/buffer-pool.md` §9 words obligation 2.
- **Consequence:** obligation 2 is stated as `specs/scheduler-design.md` §3.1 states it: the groups that stamp a request are fixed by the owner (its group, and the groups that read the class), and each stamps whether or not its table still holds the entries. `KvCore` carries it as `StampGroups`.

## 3. The core with obligation 2 as restated (`MC_Core`), 2026-11-12

- **Model:** as entry 2, with `StampGroups`.
- **Prediction:** the search completes with no violation of `TypeOK`, `P1`, `P2`, `P3`.
- **Outcome:** INCONCLUSIVE. The search was stopped after 13 minutes at depth 25 with 139,710,503 distinct states found, 35,931,189 on the queue and no violation.
- **Consequence:** the model gains a reduction, `Eager`: a completion is handed to its executor when it is produced unless a stamped request under way names its entry. An executor's table then holds at every moment at least what it holds when completions are handed on later or dropped, so a batch addresses at least the same units and every violation with late hand-off remains one. Units, slot rows and owners are symmetric in `MC_Core`.

## 4. The core, reduced (`MC_Core` with `Eager` and symmetry), 2026-11-12

- **Model:** as entry 3, with `Eager = TRUE` and the symmetry of units, slot rows and owners.
- **Prediction:** the search completes with no violation of `TypeOK`, `P1`, `P2`, `P3`.
- **Outcome:** as predicted. The search completed: 113,917,775 states generated, 22,754,006 distinct, depth 46, no violation, in 1 min 54 s on 32 workers.

## 5. The rules of the core, each left out (`MC_Core`, `RULES=NoRn`), 2026-11-12

The invariants are checked by name; P1 is `FreeUnheld`, `Exclusive`, `WriteIsolation`, `ReadIsolation`; P2 is `KeyStable`, `PinsCached`, `OneKey`, `SharedUnwritten`; P3 is `NoLeak`, `Quiescent`. TLC reports the first violated invariant on a shortest behaviour, so where two are named either may be reported.

| Rule left out | Predicted violation | Expected behaviour |
| --- | --- | --- |
| R1 | P3 `NoLeak` | An owner that is attached is attached again; its first slot row is neither free nor held |
| R2 | P1 `FreeUnheld` | A block is committed and its owner's pin released; the unit returns to the region while the cache holds it |
| R3 | P1 `WriteIsolation` or `ReadIsolation` | A `Release` takes effect while the batch that addresses the block is in flight |
| R4 | P2 `KeyStable` | A second block is committed under a cached key and replaces the first, which is still pinned |
| R6 | P2 `SharedUnwritten` or P1 `WriteIsolation` | A commit waits; its position is released and allocated again; the commit comes due and shares the new block, which a batch writes |
| R7 | P2 `KeyStable` | `Evict` takes a block its owner pins: the pinned block has no key |

R5, R8, R9 and R10 guard requests the core does not have; they are predicted with the model that adds them.

- **Outcome:** R1 `NoLeak` in 3 states, R2 `FreeUnheld` in 9, R3 `WriteIsolation` in 8, R4 `KeyStable` in 13, R7 `KeyStable` in 9: as predicted. R6: not as predicted, `NoLeak` in 11 states. The mutation as modelled kept the waiting commit's reference to its block while choosing the block to share by position, so the held block was dropped without being reclaimed: an artefact of the mutation, not the behaviour of a commit that names its position alone.
- **Consequence:** the mutation of R6 is restated: without the rule a waiting commit holds no block. The prediction for R6 stands and is run again as entry 6.

## 6. R6 left out, restated (`MC_Core`, `RULES=NoR6`), 2026-11-12

- **Prediction:** P2 `SharedUnwritten` or P1 `WriteIsolation`, by the behaviour of entry 5.
- **Outcome:** as predicted: `WriteIsolation` in 12 states.

## 7. Copies and the host tier (`MC_Copies`), 2026-11-12

- **Model:** `KvCore` with `Clone`, `Load`, `Store`, `EvictHost`, `Finish` from the executor and as the scheduler's give-up, and R5. With no copies and no host units the model has the states of entry 4 (22,754,006 distinct, run again after the change). Two units, one slot row, one host unit, one key, two owners, one group, two batches, one request under way besides the executor's `Finish`, two copies.
- **Prediction:** the search completes with no violation.
- **Prediction, R5 left out:** the cached block becomes an owner's private block: P1 `Exclusive` and P2 `KeyStable` are false in the same state; TLC reports the first listed.
- **Outcome, R5 left out:** as predicted: `Exclusive` in 11 states.
- **Outcome, all rules:** not as predicted: `KeyStable` in 12 states, `Evict` taking a block a `Store` was reading. The cause was the model's text, not the design: the two conjuncts of `Evictable`'s guard for R2 were misaligned, so the second was read inside the first's quantifier and a copy's hold was not checked. The guard is corrected and the configuration run again as entry 8.

## 8. Copies and the host tier, guard corrected (`MC_Copies`), 2026-11-12

- **Prediction:** the search completes with no violation. `MC_Core` is unaffected by the correction (it has no copies) and is not run again for it.
- **Outcome:** as predicted. The search completed: 27,997,818 states generated, 6,648,697 distinct, depth 51, no violation, in 34 s.
