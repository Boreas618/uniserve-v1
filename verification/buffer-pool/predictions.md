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

## 15. Two epochs of one layout and the trial (`MC_Trial`), 2026-11-12

- **Model:** validity is by device memory (`GroupMemory`): the groups of two epochs of one layout map the same memory, so a block one epoch committed is valid for the other. An epoch may be on trial: what it commits is visible to it alone (`vis`) until `CommitUpdate`; `Rollback` needs its owners gone and leaves its blocks visible to it alone; R1 refuses an `Attach` for a rolled-back epoch. Property P5 `TrialVisibility` is added: a block visible to one epoch is pinned and copied only for that epoch. Configuration: a serving epoch and a target on trial, one group and one owner each over the same memory, two units, one key, one batch per group, one stamped request under way.
- **Prediction, all rules:** the search completes with no violation; the serving epoch's owner pinning a block the target committed, after the update is committed, is reached.
- **Prediction, R8 left out:** P5 `TrialVisibility`: the serving epoch's owner pins a block the target committed while it is on trial. None of P1 to P3 is violated first on a shortest behaviour.
- **Outcome:** as predicted. All rules: complete, 36,713,471 states generated, 7,388,077 distinct, depth 48, no violation, 37 s; the witness is reached. R8 left out: `TrialVisibility`.

## 16. Regions of two layouts and `Resize` (`MC_Resize`), 2026-11-12

- **Model:** a device unit belongs to a region and occupies a chunk of device memory; units of two regions may occupy the same chunk. `member` is the units in their regions; `Shrink` takes a free unit out of its region and `Grow` puts a unit in, under R10 only when no unit of a region occupies its chunk. An owner allocates in the region of its scheduler. `NoLeak` requires a unit outside its region to be neither free nor held, and `OneRegion` (P3) that no two units in regions occupy one chunk. Configuration: a serving epoch with a region of two units and a target epoch of another layout whose region starts empty over the same two chunks; one group, one slot row and one owner each; one batch per group; one stamped request under way.
- **Prediction, all rules:** the search completes with no violation; both owners holding a private block at once is reached.
- **Prediction, R10 left out:** P3 `OneRegion`: a unit of the target's region grows over a chunk a unit of the serving region still occupies.
- **Outcome:** as predicted. All rules: complete, 73,399,823 states generated, 16,278,793 distinct, depth 56, no violation, 1 min 26 s; the witness is reached. R10 left out: `OneRegion`.

## 17. The tiers, and the loss of an engine (`MC_Tiers`, `MC_Loss`), 2026-11-12

- **Model:** tasks of the tiers: `Prefetch` from the remote store or a peer into a new host block, `Store` to the remote store by the connector's task, a deadline after which the request is answered, and the peer letting go of its pin; `Save` and `Restore` as copies between an owner's slot and a host block; the loss of an engine (A7): its batches and copies are over, its waiting commits and transfers are dropped, its owners are detached. `SourcesHeld` joins P2: no copy is cached from a source that was not held throughout its read. `Hazards` switches on behaviour of the system as built; H2 caches what a prefetch read whatever its peer did. `MC_Trial` was run after the change without a written prediction, as a check that the model still parses, and completed with no violation (8,552,255 distinct states; more than in entry 15 because `Shrink` and `Grow` now apply to it).
- **Prediction, `MC_Tiers`, all rules:** the search completes with no violation; a key acknowledged by the remote store, a prefetched host copy and a restore that came due are reached.
- **Prediction, `MC_Tiers`, R9 left out:** P1 `TaskIsolation`: a prefetch passes its deadline, its host block is released, and the task still writes it.
- **Prediction, `MC_Tiers`, hazard H2:** P2 `SourcesHeld`: a prefetch from a peer caches its block after the peer let go of its pin.
- **Prediction, `MC_Loss`, all rules:** the search completes with no violation; the surviving epoch's owner pinning a block the lost epoch committed is reached.
- **Outcome, `MC_Tiers`, R9 left out:** as predicted: `TaskIsolation`. **Hazard H2:** as predicted: `SourcesHeld`. **Witnesses:** all three of `MC_Tiers` and that of `MC_Loss` are reached.
- **Outcome, `MC_Tiers`, all rules:** not as predicted: `NoLeak` in 12 states. An owner saves its trajectory and is detached while the copy runs; when the `Finish` takes effect the slot row, which only the copy still held, is neither free nor held. The cause was the model's text: the branch that caches a stored or saved host block did not let go of what the copy read. Corrected; run again as entry 18.

## 18. The tiers, corrected, and the loss of an engine (`MC_Tiers`, `MC_Loss`), 2026-11-12

- **Prediction:** both searches complete with no violation.
- **Outcome, `MC_Tiers`:** as predicted: complete, 184,027,175 states generated, 32,587,250 distinct, depth 44, no violation, 3 min 11 s.

## 19. The whole suite on the model with tiers and losses (`suite.sh`), 2026-11-12

- **Prediction:** every configuration of the design (`MC_Core`, `MC_Copies`, `MC_Transfer`, `MC_Trial`, `MC_Resize`, `MC_Tiers`, `MC_Loss`) completes with no violation; each rule left out and hazard H2 report the invariants `suite.sh` states for them.
- **Outcome:** INCONCLUSIVE for four configurations. Every mutation and hazard H2 reported the stated invariant, and `MC_Loss` (3,205,242 distinct states), `MC_Trial` (8,552,255) and `MC_Resize` (16,278,793) completed with no violation. The run was stopped while `MC_Core`, `MC_Copies`, `MC_Transfer` and `MC_Tiers` were still searching, without a violation: `Shrink` and `Grow` apply to every unit of every configuration since entry 16 and multiply their states.
- **Consequence:** a constant `Movable` names the units `Resize` may move; it is empty except in `MC_Resize`.

## 20. The hazards of the system as built (`suite.sh hazards`), 2026-11-12

Each hazard of `specs/kv-cache-s0.md` §6 is a switch of `KvCore` that replaces the designed behaviour by the behaviour that document describes. H6 and H13 are violations of P4 and are not in this entry.

| Hazard | As modelled | Configuration | Predicted violation |
| --- | --- | --- | --- |
| H1 | A `Detach` releases what the owner's transfer holds while its task runs | `MC_Transfer` | P1 `TaskIsolation` |
| H2 | A prefetch caches what it read whatever its peer did | `MC_Tiers` | P2 `SourcesHeld` |
| H3 | The end of a failed copy takes effect without waiting for its stamp | `MC_Copies` | P1 `CopyIsolation` |
| H4 | A key is taken to be on the host when its store is asked | `MC_Tiers` | P6 `Exactness` |
| H5 | A `Detach` lets go of what the owner's copies hold while they run | `MC_Copies` | P1 `CopyIsolation` or `FreeUnheld` |
| H7 | The session of a request is not checked | `MC_Trial` | P1 `WriteIsolation` or `ReadIsolation` |
| H8 | The executor's rows are never cleared | `MC_Core` | P1 `WriteIsolation` or `ReadIsolation` |
| H9 | An epoch that adopted another's cache evicts what the other epoch pins | `MC_Trial` | P1 `Exclusive`, `WriteIsolation` or `ReadIsolation` |
| H10 | A block is shared while its writer still appends | `MC_Core` | P1 `WriteIsolation` or P2 `SharedUnwritten` |
| H11 | An `Evict` is answered as done and the block stays cached | `MC_Core` | P6 `Exactness` |
| H12 | State kept twice: the free list misses an evicted block | `MC_Core` | P3 `NoLeak` |

- **Outcome:** as predicted for all eleven: H1 `TaskIsolation`, H2 `SourcesHeld`, H3 `CopyIsolation`, H4 `Exactness`, H5 `CopyIsolation`, H7 `WriteIsolation`, H8 `WriteIsolation`, H9 `Exclusive`, H10 `WriteIsolation`, H11 `Exactness`, H12 `NoLeak`. The thirteen goals of `addresses.smt2` are `unsat` (z3 4.8.9).

## 21. The design and the mutations on the model with hazard switches (`suite.sh design mutations`), 2026-11-12

- **Prediction:** the seven configurations complete with no violation and the eleven mutations report the invariants `suite.sh` states.
- **Outcome:** INCONCLUSIVE, stopped on purpose after the eleven mutations reported the stated invariants and `MC_Loss` completed (2,754,061 distinct states): reading the model for its inductive invariant showed that one visibility per key is wrong once a key has copies on two tiers, and the model changes again (entry 22).

## 22. Visibility per copy (`MC_Trial`, `MC_TrialTiers`), 2026-11-12

- **Model:** a key's device copy and its host copy each have a visibility of their own, set when the copy is made by the epoch that makes it; a prefetched host copy is one every epoch may use; an epoch on trial fills no memory of a device block others may use (it loads into a block of its own, which stays private when the key already has a device block), and its host copies do not go to the remote store. `MC_TrialTiers` is `MC_Trial` with one host unit and two copies. The switches H6 (no deadline) and H13 (no give-up of a copy) are added for the liveness checks.
- **Prediction:** `MC_Trial` and `MC_TrialTiers` complete with no violation; with R8 left out both report `TrialVisibility`.

## 23. Finality as a checked property (`MC_Finality`), 2026-11-12

- **Model:** `KvFinality` adds fairness to `KvCore` (the scheduler gives a copy up, stamped requests are stamped and taken, a batch ends, a due `Finish` takes effect, a task's deadline passes) and two temporal properties: a copy in flight ends, and a request that waits on a prefetch or a remote put is answered. One unit, one slot row, one host unit, one key, one owner, one batch, one request, one copy, one task; no symmetry.
- **Prediction, design:** both properties hold.
- **Prediction, H6 (no deadline):** `TaskFinality` is violated by a task that never completes.
- **Prediction, H13 (no give-up):** `CopyFinality` is violated by a copy whose executor never returns its `Finish`.
- **Outcome:** as predicted. Design: complete, 845,591 states generated, 183,894 distinct, both properties hold. H6: `TaskFinality` violated. H13: `CopyFinality` violated.

## 24. `MC_TrialTiers`, smaller, 2026-11-12

- **Outcome of entry 22:** `MC_Trial`: as predicted, complete, 7,388,077 distinct states, no violation; R8 left out reports `TrialVisibility` on both configurations. `MC_TrialTiers` with two units and two copies: INCONCLUSIVE, stopped at 114 million distinct states with the queue still growing and no violation.
- **Model:** `MC_TrialTiers` with one unit and one copy.
- **Prediction:** the search completes with no violation; R8 left out reports `TrialVisibility`.
- **Outcome:** as predicted: complete, 88,133,355 states generated, 17,599,952 distinct, no violation; R8 left out reports `TrialVisibility`.

## 25. The whole suite on the model as it stands (`suite.sh`), 2026-11-12

- **Prediction:** the nine configurations of the design complete with no violation (`MC_Finality` with its two temporal properties); the twelve mutations and the thirteen hazards report the properties `suite.sh` states; the thirteen address goals are `unsat`.
- **Outcome:** INCONCLUSIVE, stopped on purpose: writing the inductive invariant showed two places where the model is looser than the control plane the documents describe (entry 26), and the suite is run once after them.

## 26. Stamping groups fixed at issue; a transfer hands on only unnamed entries, 2026-11-12

- **Model:** the groups that stamp a request are recorded in the request when it is issued (`sg`), as `specs/scheduler-design.md` §3.1 has the scheduler hand a request to its executors once; before, they were read from the owner's group at the moment of submission. A transfer that ends hands the target executor only the entries no stamped request under way names (A1), where before only a `Detach` under way was considered.
- **Prediction:** `MC_Transfer`, the configuration these two changes can affect most, completes with no violation; the other configurations and every mutation and hazard are run once by `suite.sh` after the inductive invariant is settled.
- **Note:** the first run after this entry searched the model before the change, because the edit had not been applied; its result (no violation) says nothing about the change and is not counted.
- **Outcome:** as predicted: `MC_Transfer` completes with no violation on the changed model: 182,674,180 states generated, 38,135,896 distinct.

## 27. The inductive invariant as an invariant (`IndInv` of `KvInductive`), 2026-11-12

- **Model:** `KvInductive` states `IndInv`: the allocation of every unit, what each deferred item, copy and task holds, what an executor's table and a completion not yet handed on may name, the stamps of requests, and what covers each unit a batch in flight addresses. Checked first on `MC_Loss` and `MC_Trial` without a written prediction: no violation (2,788,873 and 7,459,267 distinct states).
- **Prediction:** `IndInv` is an invariant of `MC_Core`, `MC_Copies`, `MC_Transfer`, `MC_Tiers`, `MC_TrialTiers` and `MC_Resize`. Whether it is inductive is a separate question, for Apalache and TLAPS.
- **Outcome so far:** `IndInv` is an invariant of `MC_Transfer` (38,135,896 distinct states), `MC_Tiers` (32,599,322), `MC_TrialTiers` (17,941,500) and `MC_Resize` (16,365,337); `MC_Core` and `MC_Copies` are still searching.

## 28. Apalache: types, the initial state, and what `IndInv` implies (`MC_Apalache`), 2026-11-12

- **Model:** `KvCore` and `KvInductive` carry Apalache's type annotations; the private holders of a unit are stated as two sets of one type each (`TableHolders`, `CopyWriters`), and one action's duplicate `UNCHANGED` is removed: neither changes the model's meaning. `IndInv` gains `TypeInv`, the values every variable takes. `MC_Apalache` fixes the constants (two epochs, three groups over two memories, two units, three slot rows, one host unit, one key, two owners, the second epoch on trial, either engine losable, both values of `Eager`) and generates any state of collections of at most three elements that satisfies `IndInv`.
- **Not predicted in writing before the runs:** the type checker passes; `Init => IndInv` holds (length 0); `IndInv => Safety` holds for the generated states (3 min 46 s); the generated predicate is satisfiable by a state with a batch, a deferred item, a request, a copy and a task (Apalache refutes `NoWork`).
- **Prediction for the inductive step** (`--init=IndInit --inv=IndInv --length=1`): I expect at least one counterexample to induction, most likely in `BatchesOk` or `RequestsOk`, whose clauses about stamps were written from the argument and never checked from an arbitrary state.
- **Outcome so far, entry 27:** `IndInv` is an invariant of `MC_Copies` (47,047,135 distinct states) and of `MC_Core` as well: of all six.
- **Outcome, entry 28, the inductive step:** as predicted, not inductive, though not where predicted: after 28 minutes Apalache reported `HandoffOk` violated by `Lose` with `Eager` false. The state before held a completion not yet handed on, for an owner of the epoch that is lost but a group of the other epoch, named by a request of the lost epoch; the loss removes the request and leaves the completion, which is then unnamed and names no live owner. `IndInv` said nothing about a named completion.
- **Consequence:** `HandoffOk` states for every completion not yet handed on that its group is of its owner's epoch, and `DeferredOk` and `TasksOk` state the same of a transfer's target.

## 29. The inductive step again (`MC_Apalache`), 2026-11-12

- **Prediction:** another counterexample to induction, in `BatchesOk`, `RequestsOk` or `TablesMapped`. The check now skips the invariants of the generated state (`search.invariantFilter=1->.*`).
- **Outcome:** as predicted, another counterexample, after 14 minutes, though again not in the conjuncts named: `CopiesOk` violated by `Attach`. The state before held a copy for an owner that was never attached; attaching it to another group breaks "an owner a copy writes for is in the copy's group". `IndInv` did not say that an owner a copy names was attached.
- **Consequence:** `AllocOwner` (a unit is allocated only to an owner that was attached) joins `IndInv`; `CopiesOk` and `DeferredOk` state it of the owner a copy or a waiting transfer names; a remote put's block is a host unit.

## 30. The inductive step, third run (`MC_Apalache`), 2026-11-12

- **Prediction:** another counterexample to induction from a state no behaviour reaches; where, I cannot say.

## 31. Outcome of entry 25: the whole suite, 2026-11-12

- **Outcome:** as predicted, on the model with stamping groups fixed at issue (entry 26) and Apalache's type annotations (entry 28): the nine configurations complete with no violation, the twelve mutations and the thirteen hazards report the stated properties, and the thirteen address goals are `unsat`. Distinct states: `MC_Core` 172,191,447; `MC_Copies` 47,047,135; `MC_Transfer` 38,135,896; `MC_Tiers` 32,599,322; `MC_TrialTiers` 17,941,500; `MC_Resize` 16,365,337; `MC_Trial` 7,459,267; `MC_Loss` 2,788,873; `MC_Finality` 188,424 with both temporal properties. The run's lines are kept in `artifacts/buffer-pool/suite-2026-11-12.txt`.
- **After the run, without a change of meaning:** `Reached` is stated without `CHOOSE` (a stamp is reached when every batch in flight of the group is later than it, or none is in flight and it is at most the last launched); the property formulas moved from `KvCore` into one module per property (`KvP1`, `KvP2`, `KvP3`, `KvP5`, `KvP6`, with `KvFinality` for P4) and are pinned (`pins.py`). Checked after each: `MC_Loss` and `MC_Finality` give the same state counts, a mutation and a hazard report the same invariants.
