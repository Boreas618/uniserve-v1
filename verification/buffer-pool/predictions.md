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

## 9. Worker groups in the model; the earlier configurations again, 2026-11-12

- **Model:** `KvCore` gains the validity of a shared block per group, tasks as holders, `Transfer` with its task, and the rule that an owner whose transfer waits or runs takes only a `Detach`. No prediction was written before `MC_Copies` was run again after the change; it was expected to pass as in entry 8.
- **Outcome:** `MC_Copies` completed with no violation: 140,773,442 states generated, 33,499,148 distinct, depth 61, in 2 min 54 s. The state count rose because `Transfer` requests are now issued (and refused, with one group).

## 10. Transfers between two groups (`MC_Transfer`), 2026-11-12

- **Model:** two groups of one epoch with one slot row each, two units, one key, two owners, one position, one batch per group, one stamped request under way, no copies.
- **Prediction, all rules:** the search completes with no violation, and an owner attached to the group it was transferred to, with its block in that group's table, is reached.
- **Prediction, R9 left out:** P1 `TaskIsolation`: an owner is detached while its transfer's task runs, its block is reclaimed, and the task still writes it. `FreeUnheld` cannot report it, since without R9 a task is not a holder.
- **Prediction, R3 left out:** P1 `WriteIsolation` or `ReadIsolation` by a `Release`, as in entry 5; if the shortest behaviour uses the transfer instead, `TaskIsolation` cannot fail (the task holds what it writes), so one of the two is reported.
- **Outcome, all rules:** not as predicted: `ReadIsolation` in 13 states. A commit waits; its owner is transferred, and the task copies the block into the target group's memory; the commit comes due and the block is valid in the group that wrote it alone, while the owner, now in the target group, reads it there. The model recorded validity at the commit only for the writing group.
- **Outcome, R9 left out:** not as predicted: `NoLeak` in 7 states. The mutation as modelled made a task hold nothing, so the slot row it took was neither free nor held: an artefact of the mutation.
- **Outcome, witness:** an owner attached to the group it was transferred to is reached.
- **Consequence:** a block whose commit takes effect after its owner was transferred with it is valid in both groups (the transfer copied it, and nothing wrote it since the commit was stamped); `specs/buffer-pool.md` §13 states it. The mutation of R9 is restated: a task holds its blocks, and without the rule a `Detach` releases them while the task runs.

## 11. Transfers, restated (`MC_Transfer`), 2026-11-12

- **Prediction, all rules:** the search completes with no violation.
- **Prediction, R9 left out:** P1 `TaskIsolation`: the released block is free, or another owner's, while the task writes it.
- **Outcome, R9 left out:** as predicted: `TaskIsolation` in 10 states.
- **Outcome, all rules:** not as predicted: `ReadIsolation` in 16 states. As entry 10, but the owner's position is released after the transfer and before the commit comes due, so the rule of entry 10 (valid in the owner's group if the owner still holds the block) leaves the block valid in the writing group alone while a batch of the target group still reads it.
- **Consequence:** the groups in which a block is valid are the block's own, kept while it is private: the group of its owner at allocation; after a transfer the target group alone, or both groups when a commit of the block waits (the block is then read-only in both). A commit does not change them. `ReadIsolation` requires every device block a batch reads, private or shared, to be valid in the batch's group.

## 12. Transfers, validity kept per block (`MC_Transfer`), 2026-11-12

- **Prediction, all rules:** the search completes with no violation.
- **Outcome:** not as predicted: `TaskIsolation` in 16 states. A block whose commit waits is transferred to the second group and back; the second task copies it into the first group's memory, where it is still valid, and the commit comes due while the task writes it.
- **Consequence:** a task copies exactly the owner's blocks that are not valid in the target group, private or shared.

## 13. Transfers, a task copies what is not valid in the target (`MC_Transfer`), 2026-11-12

- **Prediction, all rules:** the search completes with no violation.
- **Outcome:** as predicted. The search completed: 186,819,804 states generated, 37,541,052 distinct, depth 86, no violation, in 3 min 32 s.

## 14. The earlier configurations on the model with worker groups, 2026-11-12

- **Prediction:** `MC_Core` and `MC_Copies` complete with no violation; `RULES=NoR9` on `MC_Transfer` reports `TaskIsolation`, and the mutations of R1 to R7 report a violation as before.
- **Outcome:** as predicted. `MC_Core`: complete, 773,021,171 states generated, 150,548,210 distinct, depth 52, no violation, 15 min 59 s on 48 workers. `MC_Copies`: complete, 94,751,160 generated, 22,596,078 distinct, depth 54, no violation, 2 min 35 s. Mutations: R1 `NoLeak`, R2 `FreeUnheld`, R3 `WriteIsolation`, R4 `KeyStable`, R6 `WriteIsolation`, R7 `KeyStable` on `MC_Core`; R5 `Exclusive` on `MC_Copies`; R9 `TaskIsolation` on `MC_Transfer`.
